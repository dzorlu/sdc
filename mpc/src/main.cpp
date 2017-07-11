#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["throttle"];

          vector<double> _x;
          vector<double> _y;

          // map coordinates to car coordinates
          for (int i = 0; i < ptsx.size(); i++) {
            double dx = ptsx[i] - px;
            double dy = ptsy[i] - py;
            _x.push_back(dx * cos(-psi) - dy * sin(-psi));
            _y.push_back(dx * sin(-psi) + dy * cos(-psi));
          }
          double* x = &_x[0];
          double* y = &_y[0];
          Eigen::Map<Eigen::VectorXd> xv(x, 6);
          Eigen::Map<Eigen::VectorXd> yv(y, 6);
          // third degree polynominal for target values
          auto coeffs = polyfit(xv, yv, 3);
          // x, y = (0, 0) car coordinates
          double cte = polyeval(coeffs, 0);
          // psi is also zero after transformation
          double epsi = atan(coeffs[1]);

          Eigen::VectorXd state(6);
          //state << 0, 0, 0, v, cte, epsi;
          // Peoject state forward
          state[0] = v * cos(0) * 0.1;
          state[1] = v * sin(0) * 0.1;
          state[2] = (-v / 2.67) * delta * 0.1;
          state[3] = v + a * 0.1;
          state[4] = cte + v*sin(epsi)*0.1;
          state[5] = epsi - (v / 2.67) * delta * 0.1;
          auto results = mpc.Solve(state, coeffs);
          std::cout << "steer value output " << -results[0] << std::endl;
          std::cout << "throttle  output " << results[1] << std::endl;
          double steer_value = -results[0];
          double throttle_value = results[1];

          json msgJson;
          msgJson["steering_angle"] = steer_value/(deg2rad(25));
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x;
          vector<double> mpc_y;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          for (int i = 2; i < results.size(); i ++) {
            if (i%2 == 0) {
                mpc_x.push_back(results[i]);
            }
            else {
                mpc_y.push_back(results[i]);
            }
          }

          msgJson["mpc_x"] = mpc_x;
          msgJson["mpc_y"] = mpc_y;

          //Display the waypoints/reference line
          vector<double> next_x;
          vector<double> next_y;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          for (double i = 0; i < 100; i += 3){
            next_x.push_back(i);
            next_y.push_back(polyeval(coeffs, i));
          }

          msgJson["next_x"] = next_x;
          msgJson["next_y"] = next_y;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
