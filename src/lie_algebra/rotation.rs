use std::{
    fmt::{Display, Formatter},
    ops::Mul,
};

use nalgebra::{Matrix3, RealField, Vector3};

use super::{Adjoint, LieAlgebra, Manifold};

/// ## Lie algebra(SO(3))
///
/// RotationLA = Rotation Lie Algebra
#[derive(Debug)]
pub struct RotationLA<T>(pub(crate) Vector3<T>);

impl<T> RotationLA<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self(Vector3::new(x, y, z))
    }

    pub fn from_vector3(v: Vector3<T>) -> Self {
        Self(v)
    }
}

impl<T> LieAlgebra for RotationLA<T>
where
    T: Copy + RealField,
{
    type Manifold = Rotation<T>;

    fn exp(&self) -> Rotation<T> {
        if super::approx_zero(self.0.norm()) {
            Rotation::identity()
        } else {
            let (w, angle) = super::axis_angle(&self.0);
            let w_so3 = super::hat(&w);
            Rotation(
                Matrix3::identity()
                    + w_so3 * angle.sin()
                    + w_so3 * w_so3 * (T::one() - angle.cos()),
            )
        }
    }
}

#[derive(Debug)]
pub struct RotationAdjoint<T>(Matrix3<T>);

impl<T> Adjoint for RotationAdjoint<T>
where
    T: RealField + Copy,
{
    type LieAlgebra = RotationLA<T>;

    fn act(&self, other: &Self::LieAlgebra) -> Self::LieAlgebra {
        RotationLA(self.0 * other.0)
    }
}

#[derive(Debug, Clone)]
pub struct Rotation<T>(pub(super) Matrix3<T>);

impl<T> Default for Rotation<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Display + RealField> Display for Rotation<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> Rotation<T>
where
    T: RealField,
{
    pub fn identity() -> Self {
        Self(Matrix3::identity())
    }

    pub fn from_matrix3(m: Matrix3<T>) -> Self {
        Self(m)
    }
}

impl<T> Manifold for Rotation<T>
where
    T: RealField + Copy,
{
    type LieAlgebra = RotationLA<T>;
    type Adjoint = RotationAdjoint<T>;
    type Point = Vector3<T>;
    fn log(&self) -> Self::LieAlgebra {
        fn approx_zero<T: RealField>(v: T) -> bool {
            v < T::default_epsilon()
        }

        let rot = self.0;
        let one: T = T::one();
        let two = one + one;
        let cos = (rot.trace() - one) / two;
        if cos >= one {
            RotationLA(Vector3::zeros())
        } else if cos <= -one {
            let res;
            if approx_zero(one + rot[(2, 2)]) {
                res = Vector3::from_column_slice(&[rot[(0, 2)], rot[(1, 2)], one + rot[(2, 2)]])
                    / (two * (one + rot[(2, 2)])).sqrt();
            } else if approx_zero(one + rot[(1, 1)]) {
                res = Vector3::from_column_slice(&[rot[(0, 1)], one + rot[(1, 1)], rot[(2, 1)]])
                    / (two * (one + rot[(1, 1)])).sqrt();
            } else {
                res = Vector3::from_column_slice(&[rot[(0, 0)], rot[(1, 0)], one + rot[(2, 0)]])
                    / (two * (one + rot[(0, 0)])).sqrt();
            }
            return RotationLA(res * T::pi());
        } else {
            let theta = cos.acos();
            let a = (rot - rot.transpose()) * (theta / two / theta.sin());
            RotationLA(Vector3::new(a[(2, 1)], a[(0, 2)], a[(1, 0)]))
        }
    }

    // https://ethaneade.com/lie.pdf
    fn adjoint(&self) -> RotationAdjoint<T> {
        RotationAdjoint(self.0)
    }

    fn inv(&self) -> Self {
        Self(self.0.transpose())
    }

    fn mat_mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn act(&self, other: &Vector3<T>) -> Vector3<T> {
        self.0 * other
    }
}

impl<T: RealField + Copy> Mul<&Self> for Rotation<T> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        self.mat_mul(other)
    }
}

impl<T: RealField + Copy> Mul<Self> for Rotation<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.mat_mul(&other)
    }
}

impl<T: RealField + Copy> Mul<&Vector3<T>> for Rotation<T> {
    type Output = Vector3<T>;
    fn mul(self, other: &Vector3<T>) -> Self::Output {
        self.act(other)
    }
}

impl<T: RealField + Copy> Mul<Vector3<T>> for Rotation<T> {
    type Output = Vector3<T>;
    fn mul(self, other: Vector3<T>) -> Self::Output {
        self.act(&other)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rotation_lie_algebra_construct() {
        let rotationla = RotationLA::new(1., 2., 3.);
        assert_eq!(rotationla.0[0], 1.);
        assert_eq!(rotationla.0[1], 2.);
        assert_eq!(rotationla.0[2], 3.);
    }

    #[test]
    fn test_lie_algebra_exp() {
        use std::f64::consts::FRAC_PI_2;
        let rla = RotationLA::new(FRAC_PI_2, 0., 0.);
        let se3 = rla.exp();
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

        let rla = RotationLA::new(0., 0., 0.);
        let se3 = rla.exp();
        assert_eq!(se3.0[(0, 0)], 1.);
        assert_eq!(se3.0[(0, 1)], 0.);
        assert_eq!(se3.0[(0, 2)], 0.);
        assert_eq!(se3.0[(1, 0)], 0.);
        assert_eq!(se3.0[(1, 1)], 1.);
        assert_eq!(se3.0[(1, 2)], 0.);
        assert_eq!(se3.0[(2, 0)], 0.);
        assert_eq!(se3.0[(2, 1)], 0.);
        assert_eq!(se3.0[(2, 2)], 1.);
    }

    #[test]
    fn test_rotation_construct() {
        let rotation = Rotation::<f64>::default();
        assert_eq!(rotation.0[(0, 0)], 1.);
        assert_eq!(rotation.0[(0, 1)], 0.);
        assert_eq!(rotation.0[(0, 2)], 0.);
        assert_eq!(rotation.0[(1, 0)], 0.);
        assert_eq!(rotation.0[(1, 1)], 1.);
        assert_eq!(rotation.0[(1, 2)], 0.);
        assert_eq!(rotation.0[(2, 0)], 0.);
        assert_eq!(rotation.0[(2, 1)], 0.);
        assert_eq!(rotation.0[(2, 2)], 1.);
    }

    #[test]
    fn test_rotation_display() {
        let r = Rotation::<f64>::identity();
        println!("{}", r);
    }

    #[test]
    fn test_rotation_log() {
        let rot = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., -1., 0., 1., 0.,
        ]));
        let rla = rot.log();
        assert_eq!(rla.0[0], std::f64::consts::FRAC_PI_2);
        assert_eq!(rla.0[1], 0.);
        assert_eq!(rla.0[2], 0.);
        
    }

    #[test]
    fn test_rotation_inv() {
        let rot = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., -1., 0., 1., 0.,
        ]));
        let rot_inv = rot.inv();
        assert_eq!(rot_inv.0[(0, 0)], 1.);
        assert_eq!(rot_inv.0[(0, 1)], 0.);
        assert_eq!(rot_inv.0[(0, 2)], 0.);
        assert_eq!(rot_inv.0[(1, 0)], 0.);
        assert_eq!(rot_inv.0[(1, 1)], 0.);
        assert_eq!(rot_inv.0[(1, 2)], 1.);
        assert_eq!(rot_inv.0[(2, 0)], 0.);
        assert_eq!(rot_inv.0[(2, 1)], -1.);
        assert_eq!(rot_inv.0[(2, 2)], 0.);
    }

    #[test]
    fn test_rotation_mat_mul() {
        let rot1 = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., -1., 0., 1., 0.,
        ]));
        let rot2 = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., 1., 0., -1., 0.,
        ]));
        let rotation = rot1 * rot2;
        assert_eq!(rotation.0[(0, 0)], 1.);
        assert_eq!(rotation.0[(0, 1)], 0.);
        assert_eq!(rotation.0[(0, 2)], 0.);
        assert_eq!(rotation.0[(1, 0)], 0.);
        assert_eq!(rotation.0[(1, 1)], 1.);
        assert_eq!(rotation.0[(1, 2)], 0.);
        assert_eq!(rotation.0[(2, 0)], 0.);
        assert_eq!(rotation.0[(2, 1)], 0.);
        assert_eq!(rotation.0[(2, 2)], 1.);
    }

    #[test]
    fn test_rotation_act() {
        let rot = Rotation(Matrix3::from_row_slice(&[
            1., 0., 0., 0., 0., -1., 0., 1., 0.,
        ]));
        let p = Vector3::new(1., 2., 3.);
        let p1 = rot.act(&p);
        assert_eq!(p1[0], 1.);
        assert_eq!(p1[1], -3.);
        assert_eq!(p1[2], 2.);
        let p2 = rot * p;
        assert_eq!(p2[0], 1.);
        assert_eq!(p2[1], -3.);
        assert_eq!(p2[2], 2.);
    }
}
