////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/io/OBJReader.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/optimization/parametrization/Parametrizations.hpp>
#include <polyfem/optimization/parametrization/SplineParametrizations.hpp>

#include <iostream>
#include <fstream>
#include <catch2/catch_all.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace solver;
using namespace polysolve;

#if defined(__linux__)

void verify_apply_jacobian(Parametrization &parametrization, const Eigen::VectorXd &y, bool print_grads = false)
{
	Eigen::VectorXd x = parametrization.inverse_eval(y);

	Eigen::MatrixXd dydx(x.size(), y.size());
	double eps = 1e-7;
	for (int i = 0; i < x.size(); ++i)
	{
		Eigen::VectorXd x_ = x;
		x_(i) += eps;
		auto y_plus = parametrization.eval(x_);
		x_(i) -= 2 * eps;
		auto y_minus = parametrization.eval(x_);
		auto fd = (y_plus - y_minus) / (2 * eps);
		dydx.row(i) = fd;
	}

	for (int i = 0; i < y.size(); ++i)
	{
		Eigen::VectorXd grad_y;
		grad_y.setZero(y.size());
		grad_y(i) = 1;

		Eigen::VectorXd grad_x;
		grad_x = parametrization.apply_jacobian(grad_y, x);

		if (print_grads)
			logger().trace("{.16}", grad_x.norm());
		REQUIRE((grad_x - (dydx * grad_y)).norm() < 1e-8);
	}
}

TEST_CASE("bbw-test", "[parametrization]")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi E, F;
	const std::string mesh_path = POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/simple/circle/circle140.obj");
	io::OBJReader::read(mesh_path, V, E, F);
	V.conservativeResizeLike(Eigen::MatrixXd::Zero(V.rows(), 3));

	BoundedBiharmonicWeights2Dto3D lbs_with_bbw(5, V.rows(), V, F);

	Eigen::VectorXd y = utils::flatten(V);
	verify_apply_jacobian(lbs_with_bbw, y);
}

TEST_CASE("stable-nh-v1-to-legacy", "[parametrization]")
{
	StableNHV1ToLegacy map;
	Eigen::VectorXd physical(4);
	physical << 7.3, 4.2, 2.1, 1.7;

	const Eigen::VectorXd legacy = map.eval(physical);
	REQUIRE(legacy(0) == Catch::Approx(physical(0) + 3.0 * physical(2) / 8.0 - 1e-4));
	REQUIRE(legacy(1) == Catch::Approx(physical(1) + 3.0 * physical(3) / 8.0 - 1e-4));
	REQUIRE(legacy(2) == Catch::Approx(3.0 * physical(2) / 4.0));
	REQUIRE(legacy(3) == Catch::Approx(3.0 * physical(3) / 4.0));
	REQUIRE((map.inverse_eval(legacy) - physical).norm() < 1e-12);
	verify_apply_jacobian(map, legacy);
}

#endif
