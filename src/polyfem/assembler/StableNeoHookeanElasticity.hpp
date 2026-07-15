#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	// GIPC's Stable NHK1 uses these rates instead of the Lamé coefficients
	// directly. The small volume regularizer is part of GIPC's parameter map.
	inline double stable_nhk_length_rate(const double mu)
	{
		return 4.0 * mu / 3.0;
	}

	inline double stable_nhk_volume_rate(const double lambda, const double mu)
	{
		return lambda + 5.0 * mu / 6.0 + 1e-4;
	}

	template <typename Derived>
	Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> stable_nhk_cofactor(
		const Eigen::MatrixBase<Derived> &F)
	{
		assert(F.rows() == 3 && F.cols() == 3);
		using T = typename Derived::Scalar;

		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> C(3, 3);
		C(0, 0) = F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1);
		C(0, 1) = F(1, 2) * F(2, 0) - F(1, 0) * F(2, 2);
		C(0, 2) = F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0);
		C(1, 0) = F(2, 1) * F(0, 2) - F(2, 2) * F(0, 1);
		C(1, 1) = F(2, 2) * F(0, 0) - F(2, 0) * F(0, 2);
		C(1, 2) = F(2, 0) * F(0, 1) - F(2, 1) * F(0, 0);
		C(2, 0) = F(0, 1) * F(1, 2) - F(0, 2) * F(1, 1);
		C(2, 1) = F(0, 2) * F(1, 0) - F(0, 0) * F(1, 2);
		C(2, 2) = F(0, 0) * F(1, 1) - F(0, 1) * F(1, 0);
		return C;
	}

	template <typename Derived>
	typename Derived::Scalar stable_nhk_energy(
		const Eigen::MatrixBase<Derived> &F,
		const double lambda,
		const double mu)
	{
		using T = typename Derived::Scalar;
		const double length_rate = stable_nhk_length_rate(mu);
		const double volume_rate = stable_nhk_volume_rate(lambda, mu);
		const T I2 = (F * F.transpose()).eval().trace();
		const T J = F(0, 0) * (F(1, 1) * F(2, 2) - F(1, 2) * F(2, 1)) -
			F(0, 1) * (F(1, 0) * F(2, 2) - F(1, 2) * F(2, 0)) +
			F(0, 2) * (F(1, 0) * F(2, 1) - F(1, 1) * F(2, 0));
		const T J_minus_1 = J - T(1.0) - T(length_rate / volume_rate);

		return T(0.5) * (length_rate * (I2 - T(3.0)) + volume_rate * J_minus_1 * J_minus_1);
	}

	inline Eigen::MatrixXd stable_nhk_stress(
		const Eigen::MatrixXd &F,
		const double lambda,
		const double mu)
	{
		const double length_rate = stable_nhk_length_rate(mu);
		const double volume_rate = stable_nhk_volume_rate(lambda, mu);
		const double J = F.determinant();
		const double q = volume_rate * (J - 1.0) - length_rate;
		return length_rate * F + q * stable_nhk_cofactor(F);
	}

	inline Eigen::MatrixXd stable_nhk_stress_tangent(
		const Eigen::MatrixXd &F,
		const Eigen::MatrixXd &dF,
		const double lambda,
		const double mu)
	{
		const double length_rate = stable_nhk_length_rate(mu);
		const double volume_rate = stable_nhk_volume_rate(lambda, mu);
		const double J = F.determinant();
		const Eigen::MatrixXd C = stable_nhk_cofactor(F);
		const double q = volume_rate * (J - 1.0) - length_rate;
		const double dJ = (C.array() * dF.array()).sum();

		// Differentiate the polynomial cofactor directly, so this remains
		// defined at J=0 unlike an inverse-based expression.
		Eigen::MatrixXd dC(3, 3);
		auto epsilon = [](const int i, const int j, const int k) {
			if (i == j || i == k || j == k)
				return 0.0;
			if ((i == 0 && j == 1 && k == 2) ||
				(i == 1 && j == 2 && k == 0) ||
				(i == 2 && j == 0 && k == 1))
				return 1.0;
			return -1.0;
		};
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				dC(i, j) = 0.0;
				for (int a = 0; a < 3; ++a)
					for (int b = 0; b < 3; ++b)
						for (int c = 0; c < 3; ++c)
							for (int d = 0; d < 3; ++d)
								dC(i, j) += 0.5 * epsilon(i, a, b) * epsilon(j, c, d) *
									(dF(a, c) * F(b, d) + F(a, c) * dF(b, d));
			}

		return length_rate * dF + volume_rate * dJ * C + q * dC;
	}

	class StableNeoHookeanElasticity : public GenericElastic<StableNeoHookeanElasticity>
	{
	public:
		StableNeoHookeanElasticity() = default;

		std::string name() const override { return "StableNeoHookean"; }
		bool allow_inversion() const override { return true; }

		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::map<std::string, ParamFunc> parameters() const override;

		void compute_dstress_dmu_dlambda(
			const OptAssemblerData &data,
			Eigen::MatrixXd &dstress_dmu,
			Eigen::MatrixXd &dstress_dlambda) const override;

		void update_lame_params(const Eigen::MatrixXd &lambdas, const Eigen::MatrixXd &mus) override
		{
			params_.lambda_mat_ = lambdas;
			params_.mu_mat_ = mus;
		}

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p, const double t, const int el_id,
			const DefGradMatrix<T> &F) const
		{
			if (size() != 3)
				log_and_throw_error("StableNeoHookean is only implemented for 3D volume meshes.");

			double lambda, mu;
			params_.lambda_mu(p, p, t, el_id, lambda, mu);
			return stable_nhk_energy(F, lambda, mu);
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler
