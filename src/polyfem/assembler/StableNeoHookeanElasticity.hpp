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

	inline Eigen::Matrix<double, 9, 9> stable_nhk_hessian(
		const DefGradMatrix<double> &F,
		const double lambda,
		const double mu)
	{
		const double a = stable_nhk_length_rate(mu);
		const double b = stable_nhk_volume_rate(lambda, mu);
		const double q = b * (F.determinant() - 1.0) - a;
		const auto C = stable_nhk_cofactor(F);
		Eigen::Matrix<double, 9, 1> c;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				c(3 * i + j) = C(i, j);

		// Exact NHK1 tangent, as in Unified GIPC's __snk1_exact_hessian:
		// a I + b vec(cof(F)) vec(cof(F))^T + q Hessian(det(F)).
		// GenericElastic uses ROW-major vec(F), unlike GIPC's column-major
		// convention. Differentiate the determinant polynomial, without F^-1,
		// so singular and inverted deformations remain supported.
		Eigen::Matrix<double, 9, 9> H = b * c * c.transpose();
		H.diagonal().array() += a;
		auto cross_matrix = [](const Eigen::Vector3d &v) -> Eigen::Matrix3d {
			Eigen::Matrix3d S;
			S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
			return S;
		};
		const Eigen::Matrix3d H01 = -q * cross_matrix(F.row(2).transpose());
		const Eigen::Matrix3d H02 = q * cross_matrix(F.row(1).transpose());
		const Eigen::Matrix3d H12 = -q * cross_matrix(F.row(0).transpose());
		H.block<3, 3>(0, 3) += H01;
		H.block<3, 3>(3, 0) += H01.transpose();
		H.block<3, 3>(0, 6) += H02;
		H.block<3, 3>(6, 0) += H02.transpose();
		H.block<3, 3>(3, 6) += H12;
		H.block<3, 3>(6, 3) += H12.transpose();
		return H;
	}

	class StableNeoHookeanElasticity : public GenericElastic<StableNeoHookeanElasticity>
	{
	public:
		StableNeoHookeanElasticity() { autodiff_type_ = AutodiffType::NONE; }

		std::string name() const override { return "StableNeoHookean"; }
		bool allow_inversion() const override { return true; }

		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::map<std::string, ParamFunc> parameters() const override;

		DefGradMatrix<double> gradient(
			const RowVectorNd &p, double t, int el_id, const DefGradMatrix<double> &F) const override;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hessian(
			const RowVectorNd &p, double t, int el_id, const DefGradMatrix<double> &F) const override;

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
