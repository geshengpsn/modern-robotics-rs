use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};
use std::f64::consts::PI;

pub fn approx_zero(v: f64) -> bool {
    v < 1e-6
}

pub mod Rotation {
    use super::*;
    use nalgebra::Rotation3;
    pub type LieVec3<T> = Vector3<T>;
    pub type so3<T> = Matrix3<T>;
    pub type SO3<T> = Rotation3<T>;
    pub type RotMatrix<T> = SO3<T>;

    pub fn lie_vec_to_so3<T: nalgebra::RealField + Copy>(v: &LieVec3<T>) -> so3<T> {
        let zero = T::zero();
        Matrix3::from_row_slice(&[zero, -v[2], v[1], v[2], zero, -v[0], -v[1], v[0], zero])
    }

    pub fn so3_to_lie_vec<T: Copy>(so3: &so3<T>) -> LieVec3<T> {
        Vector3::new(so3[(2, 1)], so3[(0, 2)], so3[(1, 0)])
    }
}

mod RigidBodyMotion {}

pub fn vec3_to_so3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::from_row_slice(&[0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.])
}

pub fn so3_to_vec3(so3: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(so3[(2, 1)], so3[(0, 2)], so3[(1, 0)])
}

pub fn axis_angle_3(v: &Vector3<f64>) -> (Vector3<f64>, f64) {
    let angle = v.norm();
    (v / angle, angle)
}

pub fn vec3_to_rot3(v: &Vector3<f64>) -> Matrix3<f64> {
    if approx_zero(v.norm()) {
        Matrix3::identity()
    } else {
        let (w, theta) = axis_angle_3(v);
        let w_so3 = vec3_to_so3(&w);
        Matrix3::identity() + w_so3 * theta.sin() + w_so3 * w_so3 * (1. - theta.cos())
    }
}

pub fn rot3_to_vec3(rot: &Matrix3<f64>) -> Vector3<f64> {
    let cos = (rot.trace() - 1.) / 2.;
    if cos >= 1. {
        Vector3::zeros()
    } else if cos <= -1. {
        let res;
        if approx_zero(1. + rot[(2, 2)]) {
            res = Vector3::from_column_slice(&[rot[(0, 2)], rot[(1, 2)], 1. + rot[(2, 2)]])
                / (2. * (1. + rot[(2, 2)])).sqrt();
        } else if approx_zero(1. + rot[(1, 1)]) {
            res = Vector3::from_column_slice(&[rot[(0, 1)], 1. + rot[(1, 1)], rot[(2, 1)]])
                / (2. * (1. + rot[(1, 1)])).sqrt();
        } else {
            res = Vector3::from_column_slice(&[rot[(0, 0)], rot[(1, 0)], 1. + rot[(2, 0)]])
                / (2. * (1. + rot[(0, 0)])).sqrt();
        }
        return res * PI;
    } else {
        let theta = cos.acos();
        return so3_to_vec3(&(theta / 2. / theta.sin() * (rot - rot.transpose())));
    }
}

pub fn rp_to_trans(r: &Matrix3<f64>, p: &Vector3<f64>) -> Matrix4<f64> {
    Matrix4::from_row_slice(&[
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        p[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        p[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        p[2],
        0.,
        0.,
        0.,
        1.,
    ])
}

pub fn trans_to_rp(trans: &Matrix4<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let r = Matrix3::from_row_slice(&[
        trans[(0, 0)],
        trans[(0, 1)],
        trans[(0, 2)],
        trans[(1, 0)],
        trans[(1, 1)],
        trans[(1, 2)],
        trans[(2, 0)],
        trans[(2, 1)],
        trans[(2, 2)],
    ]);
    (r, Vector3::new(trans[(0, 3)], trans[(1, 3)], trans[(2, 3)]))
}

pub fn trans_inv(trans: &Matrix4<f64>) -> Matrix4<f64> {
    let (r, p) = trans_to_rp(trans);
    rp_to_trans(&r.transpose(), &(-r.transpose() * p))
}

pub fn vec6_to_se3(v: &Vector6<f64>) -> Matrix4<f64> {
    Matrix4::from_row_slice(&[
        0., -v[2], v[1], v[3], v[2], 0., -v[0], v[4], -v[1], v[0], 0., v[5], 0., 0., 0., 0.,
    ])
}

pub fn se3_to_vec6(se3: &Matrix4<f64>) -> Vector6<f64> {
    Vector6::new(
        se3[(2, 1)],
        se3[(0, 2)],
        se3[(1, 0)],
        se3[(0, 3)],
        se3[(1, 3)],
        se3[(2, 3)],
    )
}

pub fn adjoint(trans: &Matrix4<f64>) -> Matrix6<f64> {
    let (r, p) = trans_to_rp(trans);
    let p_so3 = vec3_to_so3(&p);
    let mut res = Matrix6::zeros();
    res.view_mut((0, 0), (3, 3)).copy_from(&r);
    res.view_mut((3, 0), (3, 3)).copy_from(&(p_so3 * r));
    res.view_mut((3, 3), (3, 3)).copy_from(&r);
    res
}

pub fn screw_to_vec6(q: &Vector3<f64>, s: &Vector3<f64>, h: f64) -> Vector6<f64> {
    let mut v = Vector6::zeros();
    v.view_mut((0, 0), (3, 1)).copy_from(s);
    v.view_mut((3, 0), (3, 1)).copy_from(&(q.cross(s) + h * s));
    v
}

pub fn se3_to_trans(se3: &Matrix4<f64>) -> Matrix4<f64> {
    let v = se3_to_vec6(se3);
    vec6_to_trans(&v)
}

pub fn vec6_to_trans(vec6: &Vector6<f64>) -> Matrix4<f64> {
    let w = Vector3::new(vec6[0], vec6[1], vec6[2]);
    let v = Vector3::new(vec6[3], vec6[4], vec6[5]);
    let (axis, theta) = axis_angle_3(&w);
    if approx_zero(theta) {
        let mut res = Matrix4::identity();
        res.view_mut((0, 3), (3, 1)).copy_from(&v);
        res
    } else {
        let mut res = Matrix4::identity();
        let w_so3 = vec3_to_so3(&axis);
        let vv = Matrix3::identity() * theta
            + w_so3 * (1. - theta.cos())
            + w_so3 * w_so3 * (theta - theta.sin());
        res.view_mut((0, 0), (3, 3)).copy_from(&vec3_to_rot3(&w));
        res.view_mut((0, 3), (3, 1)).copy_from(&(vv * v / theta));
        res
    }
}

pub fn trans_to_vec6(trans: &Matrix4<f64>) -> Vector6<f64> {
    let (r, p) = trans_to_rp(trans);
    let w = rot3_to_vec3(&r);

    if w == Vector3::zeros() {
        let mut res = Vector6::zeros();
        res.view_mut((3, 0), (3, 1)).copy_from(&p);
        res
    } else {
        let (w_, theta) = axis_angle_3(&w);
        let w_so3 = vec3_to_so3(&w_);
        let v = (Matrix3::identity() / theta - w_so3 / 2.
            + (1. / theta - 1. / (theta / 2.).tan() / 2.) * w_so3 * w_so3)
            * p;
        let mut res = Vector6::zeros();
        res.view_mut((0, 0), (3, 1)).copy_from(&w);
        res.view_mut((3, 0), (3, 1)).copy_from(&(v * theta));
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::FRAC_PI_2;
    #[test]
    fn approx_zero_test() {
        assert!(approx_zero(1e-7));
        assert!(!approx_zero(1e-6));
    }

    #[test]
    fn vec3_to_so3_test() {
        let res = vec3_to_so3(&Vector3::new(1., 2., 3.));
        println!("{}", res);
    }

    #[test]
    fn so3_to_vec3_test() {
        let res = so3_to_vec3(&vec3_to_so3(&Vector3::new(1., 2., 3.)));
        println!("{}", res);
    }

    #[test]
    fn axis_angle_3_test() {
        let res = axis_angle_3(&Vector3::new(1., 2., 3.));
        println!("{} {}", res.0, res.1);
    }

    #[test]
    fn so3_to_rot3_test() {
        let res = vec3_to_rot3(&Vector3::new(0., 0., PI / 2.));
        println!("{}", res);
    }

    #[test]
    fn rot3_to_so3_test() {
        let res = rot3_to_vec3(&vec3_to_rot3(&Vector3::new(0., 0., PI / 2.)));
        println!("{}", res);
    }

    #[test]
    fn rp_to_trans_test() {
        let res = rp_to_trans(
            &vec3_to_rot3(&Vector3::new(0., 0., PI / 2.)),
            &Vector3::new(1., 2., 3.),
        );
        println!("{}", res);
    }

    #[test]
    fn trans_to_rp_test() {
        let trans = rp_to_trans(
            &vec3_to_rot3(&Vector3::new(0., 0., 0.)),
            &Vector3::new(1., 2., 3.),
        );
        let (r, p) = trans_to_rp(&trans);
        println!("{} {}", r, p);
    }

    #[test]
    fn trans_inv_test() {
        let trans = rp_to_trans(
            &vec3_to_rot3(&Vector3::new(0., 0., 0.)),
            &Vector3::new(1., 2., 3.),
        );
        let res = trans_inv(&trans);
        println!("{}", res);
    }

    #[test]
    fn vec6_to_se3_test() {
        let res = vec6_to_se3(&Vector6::new(1., 2., 3., 4., 5., 6.));
        println!("{}", res);
    }

    #[test]
    fn se3_to_vec6_test() {
        let res = se3_to_vec6(&vec6_to_se3(&Vector6::new(1., 2., 3., 4., 5., 6.)));
        println!("{}", res);
    }

    #[test]
    fn adjoint_test() {
        let res = adjoint(&Matrix4::from_row_slice(&[
            1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 3., 0., 0., 0., 1.,
        ]));
        println!("{}", res);
    }

    #[test]
    fn screw_to_vec6_test() {
        let res = screw_to_vec6(&Vector3::new(3., 0., 0.), &Vector3::new(0., 0., 1.), 2.);
        println!("{}", res);
    }

    #[test]
    fn vec6_to_trans_test() {
        // let t = se3_to_trans(&vec6_to_se3(&Vector6::new(0., 0., 0., 1., 2., 3.)));
        // println!("{}", t);
        let t = vec6_to_trans(&se3_to_vec6(&Matrix4::from_row_slice(&[
            0., 0., 0., 0., 0., 0., -FRAC_PI_2, 2.35619449, 0., FRAC_PI_2, 0., 2.35619449, 0., 0.,
            0., 0.,
        ])));
        println!("{}", t);

        let t = vec6_to_trans(&Vector6::new(0., 0., 1., 0., -1., 0.));
        let mut m = Matrix4::identity();
        m[(0, 3)] = 2.;
        let t = t * m;
        println!("{}", t);
    }

    #[test]
    fn trans_to_vec6_test() {
        let res = trans_to_vec6(&Matrix4::from_row_slice(&[
            1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 3., 0., 0., 0., 1.,
        ]));
        println!("{}", res);
    }
}
