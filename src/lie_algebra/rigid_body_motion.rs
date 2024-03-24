use std::fmt::Display;

use crate::lie_algebra::axis_angle;

use super::{
    rotation::{Rotation, RotationLA},
    Adjoint, LieAlgebra, Manifold,
};
use nalgebra::{Matrix3, Matrix4, Matrix6, RealField, Vector3, Vector4, Vector6};

#[derive(Debug, Clone)]
pub struct RigidTransformLA<T>(Vector6<T>);

impl<T> Default for RigidTransformLA<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(Vector6::zeros())
    }
}

impl<T> Display for RigidTransformLA<T>
where
    T: RealField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> RigidTransformLA<T>
where
    T: Copy,
{
    pub fn new(rx: T, ry: T, rz: T, x: T, y: T, z: T) -> Self {
        Self(Vector6::new(rx, ry, rz, x, y, z))
    }

    pub fn w(&self) -> Vector3<T> {
        Vector3::new(self.0[0], self.0[1], self.0[2])
    }

    pub fn v(&self) -> Vector3<T> {
        Vector3::new(self.0[3], self.0[4], self.0[5])
    }
}

impl<T> LieAlgebra for RigidTransformLA<T>
where
    T: RealField + Copy,
{
    type Manifold = RigidTransform<T>;

    fn exp(&self) -> RigidTransform<T> {
        let v = self.v();
        let w = self.w();
        let (axis, theta) = super::axis_angle(&w);
        if super::approx_zero(theta) {
            let mut res = Matrix4::identity();
            res.view_mut((0, 3), (3, 1)).copy_from(&v);
            RigidTransform(res)
        } else {
            let mut res = Matrix4::<T>::identity();
            let w_so3 = super::hat(&axis);
            let vv = Matrix3::identity() * theta
                + w_so3 * (T::one() - theta.cos())
                + w_so3 * w_so3 * (theta - theta.sin());
            let rotla = RotationLA::from_vector3(w);
            let rot = rotla.exp();
            res.view_mut((0, 0), (3, 3)).copy_from(&rot.0);
            res.view_mut((0, 3), (3, 1)).copy_from(&(vv * v / theta));
            RigidTransform(res)
        }
    }
}

#[derive(Debug)]
pub struct RigidTransformAdjoint<T>(Matrix6<T>);

impl<T> RigidTransformAdjoint<T>
where
    T: RealField + Copy,
{
    pub fn from_rt(rt: &RigidTransform<T>) -> Self {
        rt.adjoint()
    }
}

impl<T> Adjoint for RigidTransformAdjoint<T>
where
    T: RealField + Copy,
{
    type LieAlgebra = RigidTransformLA<T>;

    fn act(&self, other: &Self::LieAlgebra) -> Self::LieAlgebra {
        RigidTransformLA(self.0 * other.0)
    }
}

#[derive(Debug, Clone)]
pub struct RigidTransform<T>(Matrix4<T>);

impl<T> RigidTransform<T>
where
    T: RealField + Copy,
{
    pub fn identity() -> Self {
        Self(Matrix4::<T>::identity())
    }

    pub fn from_rot_trans(rotation: &Rotation<T>, translation: &Vector3<T>) -> Self {
        let mut res = Matrix4::<T>::identity();
        res.view_mut((0, 0), (3, 3)).copy_from(&rotation.0);
        res.view_mut((0, 3), (3, 1)).copy_from(translation);
        Self(res)
    }

    pub fn to_rot_trans(&self) -> (Rotation<T>, Vector3<T>) {
        let r = Rotation::<T>(self.0.fixed_view::<3, 3>(0, 0).into_owned());
        let t = self.0.fixed_view::<3, 1>(0, 3).into_owned();
        (r, t)
    }
}

impl<T> Default for RigidTransform<T>
where
    T: RealField + Copy,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T> Display for RigidTransform<T>
where
    T: RealField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> Manifold for RigidTransform<T>
where
    T: RealField + Copy,
{
    type LieAlgebra = RigidTransformLA<T>;
    type Adjoint = RigidTransformAdjoint<T>;
    type Point = Vector4<T>;

    fn act(&self, other: &Vector4<T>) -> Vector4<T> {
        self.0 * other
    }

    fn adjoint(&self) -> RigidTransformAdjoint<T> {
        let mut adjoint = Matrix6::<T>::zeros();
        let (r, t) = self.to_rot_trans();
        adjoint.view_mut((0, 0), (3, 3)).copy_from(&r.0);
        adjoint.view_mut((3, 3), (3, 3)).copy_from(&r.0);
        adjoint
            .view_mut((3, 0), (3, 3))
            .copy_from(&(super::hat(&t) * r.0));
        RigidTransformAdjoint(adjoint)
    }

    fn inv(&self) -> Self {
        let (r, p) = self.to_rot_trans();
        Self::from_rot_trans(&r.inv(), &(-(r.inv() * p)))
    }

    fn log(&self) -> RigidTransformLA<T> {
        let mut res = Vector6::zeros();
        let (r, p) = self.to_rot_trans();
        let w = r.log();
        if super::approx_zero_vec(w.0) {
            res.view_mut((3, 0), (3, 1)).copy_from(&p);
            RigidTransformLA(res)
        } else {
            let one = T::one();
            let two = one + one;
            let (norm_w, theta) = axis_angle(&w.0);
            let w_so3 = RotationLA::from_vector3(norm_w).exp();

            let temp = one / theta - one / (theta / two).tan() / two;
            let v = (Matrix3::identity() / theta - w_so3.0 / two + w_so3.0 * w_so3.0 * temp) * p;
            res.view_mut((0, 0), (3, 1)).copy_from(&w.0);
            res.view_mut((3, 0), (3, 1)).copy_from(&(v * theta));
            RigidTransformLA(res)
        }
    }

    fn mat_mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rigid_transform_lie_algebra_construct() {
        let rtla = RigidTransformLA::new(1., 2., 3., 4., 5., 6.);
        assert_eq!(rtla.0[0], 1.);
        assert_eq!(rtla.0[1], 2.);
        assert_eq!(rtla.0[2], 3.);
        assert_eq!(rtla.0[3], 4.);
        assert_eq!(rtla.0[4], 5.);
        assert_eq!(rtla.0[5], 6.);

        let rtla = RigidTransformLA::<f64>::default();
        assert_eq!(rtla.0[0], 0.);
        assert_eq!(rtla.0[1], 0.);
        assert_eq!(rtla.0[2], 0.);
        assert_eq!(rtla.0[3], 0.);
        assert_eq!(rtla.0[4], 0.);
        assert_eq!(rtla.0[5], 0.);
    }

    #[test]
    fn test_lie_algebra_exp() {
        use std::f64::consts::FRAC_PI_2;
        let rtla = RigidTransformLA::new(FRAC_PI_2, 0., 0., 0., 0., 0.);
        let se3 = rtla.exp();
        let err = 1e-6;
        assert!((se3.0[(0, 0)] - 1.).abs() < err);
        assert!((se3.0[(0, 1)] - 0.).abs() < err);
        assert!((se3.0[(0, 2)] - 0.).abs() < err);
        assert!((se3.0[(1, 0)] - 0.).abs() < err);
        assert!((se3.0[(1, 1)] - 0.).abs() < err);
        assert!((se3.0[(1, 2)] - -1.).abs() < err);
        assert!((se3.0[(2, 0)] - 0.).abs() < err);
        assert!((se3.0[(2, 1)] - 1.).abs() < err);
        assert!((se3.0[(2, 2)] - 0.).abs() < err);
    }

    #[test]
    fn rigid_transform_adjoint_construct() {
        let rotation = Rotation::<f64>::identity();
        let rt = RigidTransform::<f64>::from_rot_trans(&rotation, &Vector3::new(1., 2., 3.));
        let rt_adjoint = RigidTransformAdjoint::<f64>::from_rt(&rt);
        assert!(rt_adjoint.0[(0, 0)] == 1.);
        assert!(rt_adjoint.0[(0, 1)] == 0.);
        assert!(rt_adjoint.0[(0, 2)] == 0.);
        assert!(rt_adjoint.0[(1, 0)] == 0.);
        assert!(rt_adjoint.0[(1, 1)] == 1.);
        assert!(rt_adjoint.0[(1, 2)] == 0.);
        assert!(rt_adjoint.0[(2, 0)] == 0.);
        assert!(rt_adjoint.0[(2, 1)] == 0.);
        assert!(rt_adjoint.0[(2, 2)] == 1.);

        assert!(rt_adjoint.0[(3, 0)] == 0.);
        assert!(rt_adjoint.0[(3, 1)] == -3.);
        assert!(rt_adjoint.0[(3, 2)] == 2.);
        assert!(rt_adjoint.0[(4, 0)] == 3.);
        assert!(rt_adjoint.0[(4, 1)] == 0.);
        assert!(rt_adjoint.0[(4, 2)] == -1.);
        assert!(rt_adjoint.0[(5, 0)] == -2.);
        assert!(rt_adjoint.0[(5, 1)] == 1.);
        assert!(rt_adjoint.0[(5, 2)] == 0.);
    }

    #[test]
    fn rigid_transform_adjoint_act() {
        let rotation = Rotation::<f64>::identity();
        let rt = RigidTransform::<f64>::from_rot_trans(&rotation, &Vector3::new(1., 2., 3.));

        let adjoint = rt.adjoint();
        let rtla = RigidTransformLA::new(1., 1., 1., 1., 1., 1.);
        let result = adjoint.act(&rtla);

        assert_eq!(result.0[0], 1.);
        assert_eq!(result.0[1], 1.);
        assert_eq!(result.0[2], 1.);
        assert_eq!(result.0[3], 0.);
        assert_eq!(result.0[4], 3.);
        assert_eq!(result.0[5], 0.);
    }

    #[test]
    fn rigid_transform_construct() {
        let rt = RigidTransform::<f64>::default();
        assert!(rt.0[(0, 0)] == 1.);
        assert!(rt.0[(0, 1)] == 0.);
        assert!(rt.0[(0, 2)] == 0.);
        assert!(rt.0[(0, 3)] == 0.);
        assert!(rt.0[(1, 0)] == 0.);
        assert!(rt.0[(1, 1)] == 1.);
        assert!(rt.0[(1, 2)] == 0.);
        assert!(rt.0[(1, 3)] == 0.);
        assert!(rt.0[(2, 0)] == 0.);
        assert!(rt.0[(2, 1)] == 0.);
        assert!(rt.0[(2, 2)] == 1.);
        assert!(rt.0[(2, 3)] == 0.);
        assert!(rt.0[(3, 0)] == 0.);
        assert!(rt.0[(3, 1)] == 0.);
        assert!(rt.0[(3, 2)] == 0.);
        assert!(rt.0[(3, 3)] == 1.);

        let rot = Rotation::<f64>::identity();
        let rt = RigidTransform::<f64>::from_rot_trans(&rot, &Vector3::new(1., 2., 3.));
        assert!(rt.0[(0, 0)] == 1.);
        assert!(rt.0[(0, 1)] == 0.);
        assert!(rt.0[(0, 2)] == 0.);
        assert!(rt.0[(0, 3)] == 1.);
        assert!(rt.0[(1, 0)] == 0.);
        assert!(rt.0[(1, 1)] == 1.);
        assert!(rt.0[(1, 2)] == 0.);
        assert!(rt.0[(1, 3)] == 2.);
        assert!(rt.0[(2, 0)] == 0.);
        assert!(rt.0[(2, 1)] == 0.);
        assert!(rt.0[(2, 2)] == 1.);
        assert!(rt.0[(2, 3)] == 3.);
        assert!(rt.0[(3, 0)] == 0.);
        assert!(rt.0[(3, 1)] == 0.);
        assert!(rt.0[(3, 2)] == 0.);
        assert!(rt.0[(3, 3)] == 1.);
    }

    #[test]
    fn rigid_transform_to_rot_trans() {
        let rt = RigidTransform::<f64>::default();
        let (rot, trans) = rt.to_rot_trans();
        assert!(rot.0[(0, 0)] == 1.);
        assert!(rot.0[(0, 1)] == 0.);
        assert!(rot.0[(0, 2)] == 0.);
        assert!(rot.0[(1, 0)] == 0.);
        assert!(rot.0[(1, 1)] == 1.);
        assert!(rot.0[(1, 2)] == 0.);
        assert!(rot.0[(2, 0)] == 0.);
        assert!(rot.0[(2, 1)] == 0.);
        assert!(rot.0[(2, 2)] == 1.);

        assert!(trans[0] == 0.);
        assert!(trans[1] == 0.);
        assert!(trans[2] == 0.);
    }

    #[test]
    fn rigid_transform_act() {
        let rot: Rotation<f64> = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., -1., 0., 1., 0.,
        ]));

        let p = rot.act(&Vector3::new(1., 2., 3.));
        assert_eq!(p[0], 1.);
        assert_eq!(p[1], -3.);
        assert_eq!(p[2], 2.);
    }

    #[test]
    fn rigid_transform_adjoint() {
        let rotation = Rotation::<f64>::identity();
        let rt = RigidTransform::<f64>::from_rot_trans(&rotation, &Vector3::new(1., 2., 3.));

        let rt_adjoint = rt.adjoint();
        assert!(rt_adjoint.0[(0, 0)] == 1.);
        assert!(rt_adjoint.0[(0, 1)] == 0.);
        assert!(rt_adjoint.0[(0, 2)] == 0.);
        assert!(rt_adjoint.0[(1, 0)] == 0.);
        assert!(rt_adjoint.0[(1, 1)] == 1.);
        assert!(rt_adjoint.0[(1, 2)] == 0.);
        assert!(rt_adjoint.0[(2, 0)] == 0.);
        assert!(rt_adjoint.0[(2, 1)] == 0.);
        assert!(rt_adjoint.0[(2, 2)] == 1.);

        assert!(rt_adjoint.0[(3, 0)] == 0.);
        assert!(rt_adjoint.0[(3, 1)] == -3.);
        assert!(rt_adjoint.0[(3, 2)] == 2.);
        assert!(rt_adjoint.0[(4, 0)] == 3.);
        assert!(rt_adjoint.0[(4, 1)] == 0.);
        assert!(rt_adjoint.0[(4, 2)] == -1.);
        assert!(rt_adjoint.0[(5, 0)] == -2.);
        assert!(rt_adjoint.0[(5, 1)] == 1.);
        assert!(rt_adjoint.0[(5, 2)] == 0.);
    }

    #[test]
    fn rigid_transform_inv() {
        let rotation = RotationLA::<f64>::new(0., 0., 0.).exp();
        let rt = RigidTransform::from_rot_trans(&rotation, &Vector3::new(1., 2., 3.));
        let rt_inv = rt.inv();
        assert!(rt_inv.0[(0, 0)] == 1.);
        assert!(rt_inv.0[(0, 1)] == 0.);
        assert!(rt_inv.0[(0, 2)] == 0.);
        assert!(rt_inv.0[(1, 0)] == 0.);
        assert!(rt_inv.0[(1, 1)] == 1.);
        assert!(rt_inv.0[(1, 2)] == 0.);
        assert!(rt_inv.0[(2, 0)] == 0.);
        assert!(rt_inv.0[(2, 1)] == 0.);
        assert!(rt_inv.0[(2, 2)] == 1.);

        assert!(rt_inv.0[(0, 3)] == -1.);
        assert!(rt_inv.0[(1, 3)] == -2.);
        assert!(rt_inv.0[(2, 3)] == -3.);
    }

    #[test]
    fn rigid_transform_log() {
        let rt = RigidTransformLA::new(0., 0., 0., 1., 2., 3.).exp();
        let rtla = rt.log();
        println!("{:?}", rtla);
        assert_eq!(rtla.0[0], 0.);
        assert_eq!(rtla.0[1], 0.);
        assert_eq!(rtla.0[2], 0.);
        assert_eq!(rtla.0[3], 1.);
        assert_eq!(rtla.0[4], 2.);
        assert_eq!(rtla.0[5], 3.);
    }

    #[test]
    fn rigid_transform_matmul() {
        let rt1 = RigidTransformLA::new(0., 0., 0., 1., 2., 3.).exp();
        let rt2 = RigidTransformLA::new(0., 0., 0., 1., 2., 3.).exp();

        let res = rt1.mat_mul(&rt2);

        assert_eq!(res.0[(0, 3)], 2.);
        assert_eq!(res.0[(1, 3)], 4.);
        assert_eq!(res.0[(2, 3)], 6.);
    }
}
