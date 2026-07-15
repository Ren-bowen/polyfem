#include <polyfem/State.hpp>

#include <polyfem/Units.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/NeoHookeanElasticityAutodiff.hpp>
#include <polyfem/assembler/StableNeoHookeanElasticity.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/RefElementSampler.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;
using namespace polyfem::utils;

TEST_CASE("hessian_lin", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/plane_hole.obj";
	in_args["geometry"]["surface_selection"] = 7;
	// in_args["geometry"]["mesh"] = path + "/circle2.msh";
	// in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["type"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "LinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	SparseMatrixCache mat_cache;
	StiffnessMatrix hessian, stiffness;
	varform::VarForm &form = *state.variational_formulation;
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(form);
	REQUIRE(debug.assembler != nullptr);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	AssemblyValsCache ass_vals_cache;
	ass_vals_cache.init_empty();
	Eigen::MatrixXd disp(debug.n_bases * debug.mesh->dimension(), 1);
	disp.setZero();

	REQUIRE(test::VarFormTestAccess::build_stiffness_mat(form, stiffness));

	for (int rand = 0; rand < 10; ++rand)
	{
		debug.assembler->assemble_hessian(
			debug.mesh->is_volume(), debug.n_bases, false,
			*debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = stiffness - hessian;
		const auto val = Catch::Approx(0).margin(1e-8);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
			{
				REQUIRE(it.value() == val);
			}
		}

		disp.setRandom();
	}
}

TEST_CASE("hessian_hooke", "[assembler]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/plane_hole.obj";
	in_args["geometry"]["surface_selection"] = 7;
	// in_args["geometry"]["mesh"] = path + "/circle2.msh";
	// in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["type"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "HookeLinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	SparseMatrixCache mat_cache;
	StiffnessMatrix hessian, stiffness;
	varform::VarForm &form = *state.variational_formulation;
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(form);
	REQUIRE(debug.assembler != nullptr);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	AssemblyValsCache ass_vals_cache;
	ass_vals_cache.init_empty();
	Eigen::MatrixXd disp(debug.n_bases * debug.mesh->dimension(), 1);
	disp.setZero();

	REQUIRE(test::VarFormTestAccess::build_stiffness_mat(form, stiffness));

	for (int rand = 0; rand < 10; ++rand)
	{
		debug.assembler->assemble_hessian(
			debug.mesh->is_volume(), debug.n_bases, false,
			*debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, disp, Eigen::MatrixXd(), mat_cache, hessian);

		const StiffnessMatrix tmp = stiffness - hessian;
		const auto val = Catch::Approx(0).margin(1e-8);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(tmp, k); it; ++it)
			{
				REQUIRE(it.value() == val);
			}
		}

		disp.setRandom();
	}
}

TEST_CASE("generic_elastic_assembler", "[assembler]")
{

	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/plane_hole.obj";
	in_args["geometry"]["surface_selection"] = 7;
	// in_args["geometry"]["mesh"] = path + "/circle2.msh";
	// in_args["force_linear_geometry"] = true;

	in_args["preset_problem"] = {};
	in_args["preset_problem"]["type"] = "ElasticExact";

	in_args["materials"] = {};
	in_args["materials"]["type"] = "LinearElasticity";
	in_args["materials"]["E"] = 1e5;
	in_args["materials"]["nu"] = 0.3;

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();

	// state.compute_mesh_stats();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	NeoHookeanAutodiff autodiff;
	NeoHookeanElasticity real;

	autodiff.set_size(2);
	real.set_size(2);

	Units units;
	units.init(state.args["units"]);
	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(*state.variational_formulation);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);

	autodiff.add_multimaterial(0, in_args["materials"], units, debug.root_path);
	real.add_multimaterial(0, in_args["materials"], units, debug.root_path);

	const int el_id = 0;
	const auto &bs = (*debug.bases)[el_id];
	const auto &gbs = (*debug.geometry_bases)[el_id];
	Eigen::MatrixXd local_pts;
	Eigen::MatrixXi f;
	regular_2d_grid(10, true, local_pts, f);

	Eigen::MatrixXd displacement(debug.n_bases, 1);

	ElementAssemblyValues vals;
	vals.compute(el_id, debug.mesh->is_volume(), bs, gbs);

	const auto &quadrature = vals.quadrature;
	const QuadratureVector da = vals.det.array() * quadrature.weights.array();

	for (int rand = 0; rand < 10; ++rand)
	{
		displacement.setRandom();

		// value
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const double ea = autodiff.compute_energy(data);
			const double e = real.compute_energy(data);

			if (std::isnan(e))
				REQUIRE(std::isnan(ea));
			else
				REQUIRE(ea == Catch::Approx(e).margin(1e-12));
		}

		// grad
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const Eigen::VectorXd grada = autodiff.assemble_gradient(data);
			const Eigen::VectorXd grad = real.assemble_gradient(data);

			for (int i = 0; i < grada.size(); ++i)
			{
				if (std::isnan(grad(i)))
					REQUIRE(std::isnan(grada(i)));
				else
					REQUIRE(grada(i) == Catch::Approx(grad(i)).margin(1e-12));
			}
		}

		// hessian
		{
			const NonLinearAssemblerData data(vals, 0, 0, displacement, displacement, da);

			const Eigen::MatrixXd hessiana = autodiff.assemble_hessian(data);
			const Eigen::MatrixXd hessian = real.assemble_hessian(data);

			for (int i = 0; i < hessiana.size(); ++i)
			{
				if (std::isnan(hessian(i)))
					REQUIRE(std::isnan(hessiana(i)));
				else
					REQUIRE(hessiana(i) == Catch::Approx(hessian(i)).margin(1e-12));
			}
		}

		// F stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::F, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::F, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// cauchy stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::CAUCHY, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::CAUCHY, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// pk1 stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK1, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK1, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}

		// pk2 stress
		{
			Eigen::MatrixXd stressa, stress;
			autodiff.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK2, stressa);
			real.compute_stress_tensor(OutputData(0, el_id, bs, gbs, local_pts, displacement), ElasticityTensorType::PK2, stress);

			for (int i = 0; i < stressa.size(); ++i)
			{
				if (std::isnan(stress(i)))
					REQUIRE(std::isnan(stressa(i)));
				else
					REQUIRE(stressa(i) == Catch::Approx(stress(i)).margin(1e-12));
			}
		}
	}
}

TEST_CASE("stable_neo_hookean_nhk1_derivatives", "[assembler][stable_neo_hookean]")
{
	const double lambda = 7.3;
	const double mu = 2.1;
	const double length_rate = stable_nhk_length_rate(mu);
	const double volume_rate = stable_nhk_volume_rate(lambda, mu);

	// Use a nonsingular deformation so the finite-difference checks cover the
	// volumetric and isochoric terms away from the reference configuration.
	Eigen::Matrix3d F;
	F << 1.1, 0.08, -0.03,
		0.02, 0.93, 0.05,
		-0.04, 0.06, 1.04;
	REQUIRE(F.determinant() > 0.0);

	const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
	const double rest_energy = 0.5 * length_rate * length_rate / volume_rate;
	REQUIRE(stable_nhk_energy(identity, lambda, mu) == Catch::Approx(rest_energy).epsilon(1e-12));
	REQUIRE(stable_nhk_stress(identity, lambda, mu).norm() == Catch::Approx(0.0).margin(1e-12));

	REQUIRE(AssemblerUtils::is_elastic_material("StableNeoHookean"));
	const auto registered = AssemblerUtils::make_assembler("StableNeoHookean");
	REQUIRE(registered != nullptr);
	REQUIRE(registered->name() == "StableNeoHookean");

	const Eigen::Matrix3d stress = stable_nhk_stress(F, lambda, mu);
	const double step = 1e-6;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
		{
			Eigen::Matrix3d F_plus = F;
			Eigen::Matrix3d F_minus = F;
			F_plus(i, j) += step;
			F_minus(i, j) -= step;
			const double finite_energy_gradient =
				(stable_nhk_energy(F_plus, lambda, mu) - stable_nhk_energy(F_minus, lambda, mu)) / (2.0 * step);
			REQUIRE(stress(i, j) == Catch::Approx(finite_energy_gradient).epsilon(1e-7).margin(1e-9));
		}

	Eigen::Matrix<double, 9, 9> hessian;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
				for (int l = 0; l < 3; ++l)
			{
				Eigen::Matrix3d direction = Eigen::Matrix3d::Zero();
				direction(k, l) = 1.0;
				const Eigen::Matrix3d tangent = stable_nhk_stress_tangent(F, direction, lambda, mu);
				hessian(i * 3 + j, k * 3 + l) = tangent(i, j);
			}

	REQUIRE((hessian - hessian.transpose()).norm() == Catch::Approx(0.0).margin(1e-10));
	for (int k = 0; k < 3; ++k)
		for (int l = 0; l < 3; ++l)
		{
			Eigen::Matrix3d F_plus = F;
			Eigen::Matrix3d F_minus = F;
			F_plus(k, l) += step;
			F_minus(k, l) -= step;
			const Eigen::Matrix3d finite_tangent =
				(stable_nhk_stress(F_plus, lambda, mu) - stable_nhk_stress(F_minus, lambda, mu)) / (2.0 * step);
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					REQUIRE(hessian(i * 3 + j, k * 3 + l) ==
						Catch::Approx(finite_tangent(i, j)).epsilon(1e-6).margin(1e-8));
		}

	const Eigen::Matrix3d expected_dmu =
		(4.0 / 3.0) * F + ((5.0 / 6.0) * (F.determinant() - 1.0) - 4.0 / 3.0) * stable_nhk_cofactor(F);
	const Eigen::Matrix3d expected_dlambda = (F.determinant() - 1.0) * stable_nhk_cofactor(F);
	const double parameter_step = 1e-6;
	const Eigen::Matrix3d finite_dmu =
		(stable_nhk_stress(F, lambda, mu + parameter_step) - stable_nhk_stress(F, lambda, mu - parameter_step)) /
		(2.0 * parameter_step);
	const Eigen::Matrix3d finite_dlambda =
		(stable_nhk_stress(F, lambda + parameter_step, mu) - stable_nhk_stress(F, lambda - parameter_step, mu)) /
		(2.0 * parameter_step);
	REQUIRE((expected_dmu - finite_dmu).norm() < 1e-8);
	REQUIRE((expected_dlambda - finite_dlambda).norm() < 1e-8);
}

TEST_CASE("stable_neo_hookean_generic_assembly", "[assembler][stable_neo_hookean]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = json({});
	in_args["geometry"] = {};
	in_args["geometry"]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-6.msh";
	in_args["materials"] = {
		{"type", "StableNeoHookean"},
		{"E", 2e4},
		{"nu", 0.3}};

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh();
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	const test::VarFormDebugData debug = test::VarFormTestAccess::debug_data(*state.variational_formulation);
	REQUIRE(debug.assembler != nullptr);
	REQUIRE(debug.assembler->name() == "StableNeoHookean");
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.mesh->is_volume());

	AssemblyValsCache ass_vals_cache;
	ass_vals_cache.init_empty();
	const int ndof = debug.n_bases * debug.mesh->dimension();
	Eigen::MatrixXd displacement = Eigen::MatrixXd::Zero(ndof, 1);
	Eigen::MatrixXd previous = displacement;

	auto energy = [&](const Eigen::MatrixXd &x) {
		return debug.assembler->assemble_energy(
			true, *debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, x, x);
	};
	auto gradient = [&](const Eigen::MatrixXd &x) {
		Eigen::MatrixXd result;
		debug.assembler->assemble_gradient(
			true, debug.n_bases, *debug.bases, *debug.geometry_bases, ass_vals_cache, 0, 0, x, x, result);
		return result;
	};
	auto hessian = [&](const Eigen::MatrixXd &x) {
		utils::SparseMatrixCache mat_cache;
		StiffnessMatrix result;
		debug.assembler->assemble_hessian(
			true, debug.n_bases, false, *debug.bases, *debug.geometry_bases,
			ass_vals_cache, 0, 0, x, x, mat_cache, result);
		return Eigen::MatrixXd(result);
	};

	const Eigen::MatrixXd rest_gradient = gradient(displacement);
	REQUIRE(rest_gradient.norm() == Catch::Approx(0.0).margin(1e-10));

	displacement.setRandom();
	displacement /= 100.0;
	const Eigen::MatrixXd analytic_gradient = gradient(displacement);
	const double step = 1e-6;
	Eigen::MatrixXd finite_gradient(ndof, 1);
	for (int i = 0; i < ndof; ++i)
	{
		Eigen::MatrixXd plus = displacement;
		Eigen::MatrixXd minus = displacement;
		plus(i) += step;
		minus(i) -= step;
		finite_gradient(i) = (energy(plus) - energy(minus)) / (2.0 * step);
	}
	REQUIRE((analytic_gradient - finite_gradient).norm() < 1e-5);

	const Eigen::MatrixXd analytic_hessian = hessian(displacement);
	Eigen::MatrixXd finite_hessian(ndof, ndof);
	for (int i = 0; i < ndof; ++i)
	{
		Eigen::MatrixXd plus = displacement;
		Eigen::MatrixXd minus = displacement;
		plus(i) += step;
		minus(i) -= step;
		finite_hessian.col(i) = (gradient(plus) - gradient(minus)) / (2.0 * step);
	}
	REQUIRE((analytic_hessian - analytic_hessian.transpose()).norm() < 1e-10);
	REQUIRE((analytic_hessian - finite_hessian).norm() < 1e-4 * std::max(1.0, analytic_hessian.norm()));
}
