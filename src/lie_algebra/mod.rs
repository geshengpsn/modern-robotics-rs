use nalgebra::{Matrix3, RealField, Vector3};

pub mod rigid_body_motion;
pub mod rotation;

pub trait LieAlgebra {
    type Manifold;
    fn exp(&self) -> Self::Manifold;
}

pub trait Adjoint {
    type LieAlgebra;
    fn act(&self, other: &Self::LieAlgebra) -> Self::LieAlgebra;
}

pub trait Manifold {
    type LieAlgebra;
    type Adjoint;
    type Point;
    // lie algebra
    fn log(&self) -> Self::LieAlgebra;
    fn adjoint(&self) -> Self::Adjoint;

    // group
    fn inv(&self) -> Self;
    fn mat_mul(&self, other: &Self) -> Self;
    fn act(&self, other: &Self::Point) -> Self::Point;
}

fn approx_zero<T: RealField>(v: T) -> bool {
    v < T::default_epsilon()
}

fn approx_zero_vec<T: RealField>(vec: Vector3<T>) -> bool {
    vec[0] < T::default_epsilon() && vec[1] < T::default_epsilon() && vec[2] < T::default_epsilon()
}

fn axis_angle<T: RealField + Copy>(v: &Vector3<T>) -> (Vector3<T>, T) {
    let angle = v.norm();
    (v / angle, angle)
}

fn hat<T>(v: &Vector3<T>) -> Matrix3<T>
where
    T: RealField + Copy,
{
    let zero = T::zero();
    Matrix3::new(zero, -v[2], v[1], v[2], zero, -v[0], -v[1], v[0], zero)
}
