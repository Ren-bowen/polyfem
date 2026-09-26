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

TEST_CASE("clamp-map matches bounded log-material mapping", "[parametrization][clamp]")
{
	ClampMap clamp(4.0, 16.0);
	Eigen::VectorXd x(5), expected(5), derivative(5);
	x << 3.0, 4.0, 10.0, 16.0, 17.0;
	expected << 4.0, 4.0, 10.0, 16.0, 16.0;
	derivative << 0.0, 1.0, 1.0, 1.0, 0.0;
	REQUIRE(clamp.eval(x).isApprox(expected));
	REQUIRE(clamp.apply_jacobian(Eigen::VectorXd::Ones(5), x).isApprox(derivative));
	REQUIRE(clamp.inverse_eval(expected).isApprox(expected));
	REQUIRE_THROWS(clamp.inverse_eval(x));
	REQUIRE_THROWS(ClampMap(16.0, 4.0));

	// Check the composed exp(clamp(x)) derivative inside and outside bounds.
	ExponentialMap exp;
	x.resize(3);
	x << 3.0, 10.0, 17.0;
	const auto gradient = clamp.apply_jacobian(
		exp.apply_jacobian(Eigen::VectorXd::Ones(3), clamp.eval(x)), x);
	for (int i = 0; i < x.size(); ++i)
	{
		auto plus = x, minus = x;
		plus(i) += 1e-5;
		minus(i) -= 1e-5;
		const double fd = (exp.eval(clamp.eval(plus))(i) - exp.eval(clamp.eval(minus))(i)) / 2e-5;
		REQUIRE(std::abs(gradient(i) - fd) <= 1e-7 * std::max(1.0, std::abs(fd)));
	}
}

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

TEST_CASE("vector clamp preserves endpoint gradients", "[parametrization][clamp]")
{
    Eigen::VectorXd lo(3), hi(3), x(3), expected(3), derivative(3);
    lo << -0.2, -2, -0.2;
    hi << 0.2, 0, 0.2;
    ClampMap clamp(lo, hi);
    x << 0.3, -0.5, 0.2;
    expected << 0.2, -0.5, 0.2;
    derivative << 0, 1, 1;
    REQUIRE(clamp.eval(x).isApprox(expected));
    REQUIRE(clamp.apply_jacobian(Eigen::VectorXd::Ones(3), x).isApprox(derivative));
    REQUIRE_THROWS(clamp.eval(Eigen::VectorXd::Zero(2)));
}

#include <polysolve/nonlinear/Solver.hpp>

TEST_CASE("ordinary LBFGS clips after line search and refreshes state", "[lbfgs-clip]")
{
    class ClippedProblem : public polysolve::nonlinear::Problem
    {
    public:
        TVector observed;
        int clips = 0;
        double value(const TVector &x) override
        {
            return std::pow(std::clamp(x[0], -0.2, 0.2) - 1.0, 2) + std::pow(std::clamp(x[1], -2.0, 0.0) + 1.5, 2);
        }
        void gradient(const TVector &x, TVector &g) override
        {
            g.resize(2);
            g[0] = x[0] >= -0.2 && x[0] <= 0.2 ? 2 * (x[0] - 1) : 0;
            g[1] = x[1] >= -2 && x[1] <= 0 ? 2 * (x[1] + 1.5) : 0;
        }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("unused"); }
        void solution_changed(const TVector &x) override { observed = x; }
        bool clip_update(const TVector &x0, TVector &x1) override
        {
            ++clips;
            x1[0] = std::clamp(x1[0], std::max(-0.2, x0[0] - 0.5), std::min(0.2, x0[0] + 0.5));
            x1[1] = std::clamp(x1[1], std::max(-2.0, x0[1] - 0.5), std::min(0.0, x0[1] + 0.5));
            return true;
        }
    } problem;
    json params = {{"solver", "L-BFGS"}, {"max_iterations", 1}, {"allow_out_of_iterations", true},
                   {"line_search", {{"method", "Backtracking"}, {"use_grad_norm_tol", 0}}}};
    auto optimizer = polysolve::nonlinear::Solver::create(params, json::object(), 1.0, polyfem::logger());
    Eigen::VectorXd x(2), expected(2);
    x << 0.2, 0;
    expected << 0.2, -0.5;
    optimizer->minimize(problem, x);
    REQUIRE(problem.clips == 1);
    REQUIRE(x.isApprox(expected));
    REQUIRE(problem.observed.isApprox(x));
}

#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>

TEST_CASE("parameter clip configuration and adjoint update", "[lbfgs-clip]")
{
    json input = json::parse(R"({
        "parameters": [{"number": 3, "initial": [0.2, 0, 0]}],
        "states": [],
        "variable_to_simulation": [{"type": "initial", "state": 0,
            "composition": [{"type": "clamp", "lower": [-0.2,-2,-0.2], "upper": [0.2,0,0.2]}]}],
        "parameter_clip": {"lower": [-0.2,-2,-0.2], "upper": [0.2,0,0.2], "max_change": 0.5}
    })");
    auto args = AdjointOptUtils::apply_opt_json_spec(input, true);
    VariableToSimulationGroup variables;
    AdjointNLProblem problem(nullptr, variables, {}, {}, args);
    Eigen::VectorXd x0(3), trial(3), expected(3);
    x0 << 0.2, 0, 0;
    trial << 1, -1.5, -0.8;
    expected << 0.2, -0.5, -0.2;
    REQUIRE(problem.clip_update(x0, trial));
    REQUIRE(trial.isApprox(expected));
    REQUIRE_FALSE(problem.clip_before_line_search());
    input["parameter_clip"]["before_line_search"] = true;
    auto pre_args = AdjointOptUtils::apply_opt_json_spec(input, true);
    AdjointNLProblem pre_problem(nullptr, variables, {}, {}, pre_args);
    REQUIRE(pre_problem.clip_before_line_search());
    input["variable_to_simulation"][0]["composition"][0]["lower"] = -5;
    input["variable_to_simulation"][0]["composition"][0]["upper"] = 5;
    REQUIRE_NOTHROW(AdjointOptUtils::apply_opt_json_spec(input, true));
}

TEST_CASE("optimization explicit LBFGS does not fall back to GD", "[lbfgs-no-fallback]")
{
    class FailedSearch : public polysolve::nonlinear::Problem
    {
    public:
        int searches = 0;
        double value(const TVector &x) override { return x.squaredNorm(); }
        void gradient(const TVector &x, TVector &g) override { g = -2*x; }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("unused"); }
        void line_search_begin(const TVector &, const TVector &) override { ++searches; }
    } problem;
    json params = {{"solver", json::array({{{"type", "L-BFGS"}, {"history_size", 3}}})},
                   {"line_search", {{"method", "Backtracking"}, {"max_step_size_iter_final", 2}}}};
    auto optimizer = AdjointOptUtils::make_nl_solver(params, json::object(), 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    REQUIRE_THROWS(optimizer->minimize(problem, x));
    REQUIRE(problem.searches == 1);
    REQUIRE(x[0] == 1.0);
}

TEST_CASE("LBFGS and GD backtrack along a preclipped update", "[lbfgs-preclip]")
{
    class PreclippedProblem : public polysolve::nonlinear::Problem
    {
    public:
        std::vector<double> trials;
        int clips = 0;
        double value(const TVector &x) override { return 1000 * std::pow(x[0] - 0.47, 2); }
        void gradient(const TVector &x, TVector &g) override { g = 2000 * (x.array() - 0.47).matrix(); }
        void hessian(const TVector &, THessian &) override { throw std::runtime_error("unused"); }
        void solution_changed(const TVector &x) override { trials.push_back(x[0]); }
        bool clip_before_line_search() const override { return true; }
        bool clip_update(const TVector &x0, TVector &x1) override
        {
            ++clips;
            x1[0] = std::clamp(x1[0], std::max(0.0, x0[0] - 0.1), std::min(1.0, x0[0] + 0.1));
            return true;
        }
    } problem;
    const bool gd = GENERATE(false, true);
    json params = {{"solver", json::array({{{"type", "L-BFGS"}, {"history_size", 6}}})},
                   {"max_iterations", 1}, {"allow_out_of_iterations", true},
                   {"line_search", {{"method", "Backtracking"}, {"use_grad_norm_tol", 0}}}};
    if (gd)
    {
        params["solver"] = json::array({{{"type", "GradientDescent"}}});
        params["line_search"]["default_init_step_size"] = 0.01;
    }
    auto optimizer = AdjointOptUtils::make_nl_solver(params, json::object(), 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.5);
    optimizer->minimize(problem, x);
    REQUIRE(problem.clips == 1);
    REQUIRE(x[0] == Catch::Approx(0.45));
    REQUIRE(std::find_if(problem.trials.begin(), problem.trials.end(), [](double v) { return std::abs(v - 0.4) < 1e-12; }) != problem.trials.end());
    for (double trial : problem.trials)
        REQUIRE(trial >= 0.4 - 1e-12);
}
