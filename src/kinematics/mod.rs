use nalgebra::RealField;

use crate::{
    lie_algebra::{rigid_body_motion::RigidTransform, LieAlgebra, Manifold},
    multi_body::JointsConfig,
};

trait Jacobian<T> {
    // S = J * theta_dot
    type LA: LieAlgebra;
    fn act(&self, jc: &JointsConfig<T>) -> Self::LA;

    // J^-1
    type InvMatrix;
    fn inv(&self) -> Self::InvMatrix;

    // J^T
    type TransposeMatrix;
    fn transpose(&self) -> Self::TransposeMatrix;
}

trait DKSolver<T> {
    type Jacob: Jacobian<T>;
    fn jacobian(&self, jc: &JointsConfig<T>) -> anyhow::Result<Self::Jacob>;
}

trait FKSolver<T> {
    type Solution: Manifold;
    fn solve(&self, jc: &JointsConfig<T>) -> anyhow::Result<Self::Solution>;
}

struct RobotFKSolver<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> RobotFKSolver<T> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> FKSolver<T> for RobotFKSolver<T>
where
    T: RealField + Copy,
{
    type Solution = RigidTransform<T>;
    fn solve(&self, jc: &JointsConfig<T>) -> anyhow::Result<Self::Solution> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::MultiBody;

    use super::*;
    #[test]
    fn fk_test() {
        let robot = MultiBody::<f64>::from_file("urdf/panda.urdf").unwrap();
        let sovler = RobotFKSolver::<f64>::new();

        // sovler.solve();
        todo!("Implement fk_space_test()")
        // V_b_top = [0 0 50 0 0 0]
        // T_ob = (I, [0, 0.02t+c, 0])
        // Ad_T_ob = [I, 0; [p] T]
        // V_O_top = Ad_T_ob * V_b_top
        // V_O_top = [0 0 50 1t 0 0]
        // A_O_top = [0 0 0 1 0 0]
    }
}
