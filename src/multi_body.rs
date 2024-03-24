use anyhow::Error;
use nalgebra::{RealField, Rotation3, Vector3};
use petgraph::prelude::*;
use std::{collections::HashMap, path::Path, vec};
use urdf_rs::{read_file, read_from_string, JointType};

use crate::lie_algebra::{
    rigid_body_motion::{RigidTransform, RigidTransformLA},
    rotation::Rotation,
    Adjoint, Manifold,
};

pub struct MultiBody<T> {
    robot: urdf_rs::Robot,
    link_map: HashMap<String, NodeIndex>,
    pub(crate) graph: DiGraph<usize, usize>,
    pub(crate) m_list: Vec<RigidTransform<T>>,
    pub(crate) s_list: Vec<RigidTransformLA<T>>,
}

type LinkJointGraph = (DiGraph<usize, usize>, NodeIndex, HashMap<String, NodeIndex>);

fn construct_link_joint_graph(urdf_robot: &urdf_rs::Robot) -> anyhow::Result<LinkJointGraph> {
    let mut link_map = HashMap::with_capacity(urdf_robot.links.len());
    let mut graph = DiGraph::with_capacity(urdf_robot.links.len(), urdf_robot.joints.len());

    urdf_robot
        .links
        .iter()
        .enumerate()
        .for_each(|(link_index, l)| {
            let link_index = graph.add_node(link_index);
            link_map.insert(l.name.clone(), link_index);
        });

    urdf_robot
        .joints
        .iter()
        .enumerate()
        .for_each(|(joint_index, j)| {
            let parent_index = *link_map.get(j.parent.link.as_str()).unwrap();
            let child_index = *link_map.get(j.child.link.as_str()).unwrap();
            graph.add_edge(parent_index, child_index, joint_index);
        });

    let random_node_index = graph
        .node_indices()
        .next()
        .ok_or(Error::msg("No link found"))?;

    let mut prev = random_node_index;
    let root = loop {
        if let Some(nx) = graph.neighbors_directed(prev, Incoming).next() {
            prev = nx;
        } else {
            break prev;
        }
    };

    Ok((graph, root, link_map))
}

fn joint_to_rot_trans<T>(joint: &urdf_rs::Joint) -> (Rotation<T>, Vector3<T>)
where
    T: RealField + Copy,
{
    let r = T::from_f64(joint.origin.rpy[0]).unwrap();
    let p = T::from_f64(joint.origin.rpy[1]).unwrap();
    let y = T::from_f64(joint.origin.rpy[2]).unwrap();
    let rot = Rotation::from_matrix3(Rotation3::from_euler_angles(r, p, y).into_inner());
    let x = T::from_f64(joint.origin.xyz[0]).unwrap();
    let y = T::from_f64(joint.origin.xyz[1]).unwrap();
    let z = T::from_f64(joint.origin.xyz[2]).unwrap();
    (rot, Vector3::new(x, y, z))
}

fn construct_multi_body<T>(urdf_robot: urdf_rs::Robot) -> anyhow::Result<MultiBody<T>>
where
    T: RealField + Copy,
{
    // let data store in vector
    let (graph, root_link_index, link_map) = construct_link_joint_graph(&urdf_robot)?;

    let mut m_list = vec![RigidTransform::<T>::identity(); urdf_robot.links.len()];
    let mut s_list = vec![RigidTransformLA::<T>::default(); urdf_robot.joints.len()];

    // compute all links zero position
    let mut bfs = Bfs::new(&graph, root_link_index);
    while let Some(nx) = bfs.next(&graph) {
        // find prev_joint and prev_link
        if let (Some(edge_index), Some(node_index)) = (
            graph.edges_directed(nx, Incoming).next(),
            graph.neighbors_directed(nx, Incoming).next(),
        ) {
            let joint_index = *graph.edge_weight(edge_index.id()).unwrap();
            let node_index = *graph.node_weight(node_index).unwrap();
            let joint = &urdf_robot.joints[joint_index];
            // let link = &urdf_robot.links[node_index];
            let (rotation, translation) = joint_to_rot_trans::<T>(joint);
            let rt = RigidTransform::from_rot_trans(&rotation, &translation);

            let s_local = match &joint.joint_type {
                JointType::Prismatic => {
                    let x = T::from_f64(joint.axis.xyz[0]).unwrap();
                    let y = T::from_f64(joint.axis.xyz[1]).unwrap();
                    let z = T::from_f64(joint.axis.xyz[2]).unwrap();
                    RigidTransformLA::new(T::zero(), T::zero(), T::zero(), x, y, z)
                }
                JointType::Revolute => {
                    let rx = T::from_f64(joint.axis.xyz[0]).unwrap();
                    let ry = T::from_f64(joint.axis.xyz[1]).unwrap();
                    let rz = T::from_f64(joint.axis.xyz[2]).unwrap();
                    RigidTransformLA::new(rx, ry, rz, T::zero(), T::zero(), T::zero())
                }
                JointType::Fixed => RigidTransformLA::default(),
                jt => {
                    return Err(Error::msg(format!("unsupported joint type {:?}", jt)));
                }
            };
            let s = rt.adjoint().act(&s_local);
            s_list[joint_index] = s;
            m_list[*graph.node_weight(nx).unwrap()] = m_list[node_index].mat_mul(&rt);
        }
    }
    Ok(MultiBody {
        robot: urdf_robot,
        graph,
        m_list,
        s_list,
        link_map,
    })
}

impl<T> MultiBody<T>
where
    T: RealField + Copy,
{
    pub fn from_string(string: &str) -> anyhow::Result<Self> {
        let urdf_robot = read_from_string(string)?;
        construct_multi_body(urdf_robot)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let urdf_robot = read_file(path)?;
        construct_multi_body(urdf_robot)
    }

    pub fn default_joints(&self) -> JointsConfig<T> {
        let mut map = HashMap::new();
        for j in self.robot.joints.iter() {
            if j.joint_type != JointType::Fixed {
                map.insert(
                    j.name.clone(),
                    Joint {
                        value: T::zero(),
                        joint_type: j.joint_type.clone(),
                        lower_limit: T::from_f64(j.limit.lower).unwrap(),
                        upper_limit: T::from_f64(j.limit.upper).unwrap(),
                    },
                );
            }
        }
        JointsConfig(map)
    }

    pub(crate) fn tip_to_base(&self, tip: &str, base: &str) -> anyhow::Result<TipToBaseResult<T>> {
        // self.graph.
        let tip_index = self
            .link_map
            .get(tip)
            .ok_or(Error::msg(format!("tip {} do not exist in robot", tip)))?;

        let base_index = self
            .link_map
            .get(base)
            .ok_or(Error::msg(format!("base {} do not exist in robot", base)))?;

        let (_cost, path) = petgraph::algo::astar(
            &self.graph,
            *base_index,
            |nx| nx == *tip_index,
            |_| 1.,
            |_| 0.,
        )
        .ok_or(Error::msg(format!("no path from {} to {}", base, tip)))?;

        let m_list = path
            .iter()
            .map(|nx| {
                let m_index = *self.graph.node_weight(*nx).unwrap();
                self.m_list[m_index].clone()
            })
            .collect::<Vec<_>>();

        let s_list = path
            .iter()
            .skip(1)
            .map(|nx| {
                let edge = self.graph.edges_directed(*nx, Incoming).next().unwrap();
                let j_index = *self.graph.edge_weight(edge.id()).unwrap();
                self.s_list[j_index].clone()
            })
            .collect::<Vec<_>>();
        Ok((m_list, s_list))
    }
}

type TipToBaseResult<T> = (Vec<RigidTransform<T>>, Vec<RigidTransformLA<T>>);

#[derive(Debug)]
pub struct JointsConfig<T>(HashMap<String, Joint<T>>);

#[derive(Debug)]
pub struct Joint<T> {
    value: T,
    joint_type: JointType,
    lower_limit: T,
    upper_limit: T,
}

impl<T> JointsConfig<T>
where
    T: Copy + RealField,
{
    pub fn set(&mut self, joint_name: &str, value: T) -> anyhow::Result<()> {
        let a = self
            .0
            .get_mut(joint_name)
            .ok_or(Error::msg(format!("{} do not exist", joint_name)))?;

        if value < a.lower_limit || value > a.upper_limit {
            Err(Error::msg(format!(
                "{joint_name} value {} is out of range [{}, {}]",
                value, a.lower_limit, a.upper_limit
            )))
        } else {
            a.value = value;
            Ok(())
        }
    }

    pub fn get_value(&self, joint_name: &str) -> Option<T> {
        self.0.get(joint_name).map(|j| j.value)
    }

    pub fn get_joint_type(&self, joint_name: &str) -> Option<&JointType> {
        self.0.get(joint_name).map(|j| &j.joint_type)
    }

    pub fn get_limit(&self, joint_name: &str) -> Option<(T, T)> {
        self.0
            .get(joint_name)
            .map(|j| (j.lower_limit, j.upper_limit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_link_joint_graph_test() {
        let urdf_robot = read_file("urdf/panda.urdf").unwrap();
        let (graph, root_node, _) = construct_link_joint_graph(&urdf_robot).unwrap();
        let root_link = graph.node_weight(root_node).unwrap();
        let root_name = urdf_robot.links[*root_link].name.as_str();
        assert_eq!(root_name, "panda_link0");

        // let tree = termtree::Tree::new(root_name);
        let mut bfs = Bfs::new(&graph, root_node);
        while let Some(nx) = bfs.next(&graph) {
            let link = graph.node_weight(nx).unwrap();
            println!("{}", urdf_robot.links[*link].name);
        }
    }

    #[test]
    fn multi_body_from_urdf_test() {
        let robot = MultiBody::<f64>::from_file("urdf/panda.urdf").unwrap();
        for (s, joint) in robot.s_list.iter().zip(robot.robot.joints) {
            println!("joint: {}", joint.name);
            println!("{}", s);
        }
    }

    #[test]
    fn default_joints_test() {
        let robot = MultiBody::<f64>::from_file("urdf/panda.urdf").unwrap();
        let joints = robot.default_joints();
        println!("{:#?}", joints)
    }

    #[test]
    fn set_get_joints_config() {
        let robot = MultiBody::<f64>::from_file("urdf/panda.urdf").unwrap();
        let mut joints = robot.default_joints();
        let res = joints.set("panda_joint1", -0.1);
        assert!(res.is_ok());
        assert_eq!(joints.get_value("panda_joint1"), Some(-0.1));

        let res = joints.set("panda_joint1", 3.);
        assert!(res.is_err());
        assert_eq!(joints.get_value("panda_joint1"), Some(-0.1));
    }

    #[test]
    fn tip_to_base_test() {
        let robot = MultiBody::<f64>::from_file("urdf/panda.urdf").unwrap();
        let res = robot.tip_to_base("panda_leftfinger", "panda_link0");
        assert!(res.is_ok());
        let (m_list, s_list) = res.unwrap();
        for (m, s) in m_list.iter().zip(s_list.iter()) {
            println!("m: {}", m);
            println!("s: {}", s);
        }
    }
}
