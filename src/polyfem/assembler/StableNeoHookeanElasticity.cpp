#include "StableNeoHookeanElasticity.hpp"

namespace polyfem::assembler
{
	DefGradMatrix<double> StableNeoHookeanElasticity::gradient(
		const RowVectorNd &p, const double t, const int el_id, const DefGradMatrix<double> &F) const
	{
		double lambda, mu;
		params_.lambda_mu(p, p, t, el_id, lambda, mu);
		return stable_nhk_stress(F, lambda, mu);
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> StableNeoHookeanElasticity::hessian(
		const RowVectorNd &p, const double t, const int el_id, const DefGradMatrix<double> &F) const
	{
		double lambda, mu;
		params_.lambda_mu(p, p, t, el_id, lambda, mu);
		return stable_nhk_hessian(F, lambda, mu);
	}

	namespace
	{
		void deformation_and_lame(
			const LameParameters &params,
			const OptAssemblerData &data,
			Eigen::MatrixXd &F,
			double &lambda,
			double &mu)
		{
			params.lambda_mu(data.local_pts, data.global_pts, data.t, data.el_id, lambda, mu);
			F = Eigen::MatrixXd::Identity(3, 3) + data.grad_u_i;
		}
	} // namespace

	void StableNeoHookeanElasticity::compute_stress_grad_multiply_mat(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &mat,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		Eigen::MatrixXd F;
		deformation_and_lame(params_, data, F, lambda, mu);
		stress = stable_nhk_stress(F, lambda, mu);
		result = stable_nhk_stress_tangent(F, mat, lambda, mu);
	}

	void StableNeoHookeanElasticity::compute_stress_grad_multiply_stress(
		const OptAssemblerData &data,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		Eigen::MatrixXd F;
		deformation_and_lame(params_, data, F, lambda, mu);
		stress = stable_nhk_stress(F, lambda, mu);
		result = stable_nhk_stress_tangent(F, stress, lambda, mu);
	}

	void StableNeoHookeanElasticity::compute_stress_grad_multiply_vect(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &vect,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		double lambda, mu;
		Eigen::MatrixXd F;
		deformation_and_lame(params_, data, F, lambda, mu);
		stress = stable_nhk_stress(F, lambda, mu);
		result.setZero(9, vect.size());

		// Match GenericElastic's AD convention on row-major vec(S):
		// row vect → contract the first index of F; column vect → the second.
		for (int a = 0; a < 3; ++a)
		{
			Eigen::Matrix3d dF = Eigen::Matrix3d::Zero();
			if (vect.rows() == 1)
				dF.col(a) = vect.transpose();
			else
				dF.row(a) = vect.transpose();
			const Eigen::MatrixXd dS = stable_nhk_stress_tangent(F, dF, lambda, mu);
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					result(i * 3 + j, a) = dS(i, j);
		}
	}

	void StableNeoHookeanElasticity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		if (size() != 3)
			log_and_throw_error("StableNeoHookean is only implemented for 3D volume meshes.");

		params_.add_multimaterial(index, params, true, units.stress(), root_path);
	}

	void StableNeoHookeanElasticity::compute_dstress_dmu_dlambda(
		const OptAssemblerData &data,
		Eigen::MatrixXd &dstress_dmu,
		Eigen::MatrixXd &dstress_dlambda) const
	{
		double lambda, mu;
		params_.lambda_mu(data.local_pts, data.global_pts, data.t, data.el_id, lambda, mu);

		const Eigen::MatrixXd F = Eigen::MatrixXd::Identity(3, 3) + data.grad_u_i;
		const Eigen::MatrixXd C = stable_nhk_cofactor(F);
		const double J = F.determinant();

		dstress_dmu = (4.0 / 3.0) * F +
			((5.0 / 6.0) * (J - 1.0) - 4.0 / 3.0) * C;
		dstress_dlambda = (J - 1.0) * C;
	}

	std::map<std::string, Assembler::ParamFunc> StableNeoHookeanElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;

		res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);
			return lambda;
		};

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu;
		};

		res["E"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
		};

		res["nu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);
			return lambda / (2.0 * (lambda + mu));
		};

		return res;
	}
} // namespace polyfem::assembler
