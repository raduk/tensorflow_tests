#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"

#include <sstream>
#include <string>
#include <fstream>

#include "tensorflow/core/framework/graph_def_util.h"
#include <chrono>

using namespace tensorflow;

const std::string GRAPH_FILE = "test.pb";

int main(int argc, char* argv[]) {
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), GRAPH_FILE, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  status = session->Run({}, {}, {"init"}, nullptr);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  std::cout << "Starting to evaluate " << std::endl;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  status = session->Run({}, {}, {"MatMul"}, nullptr);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  std::cout << "Total execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()  << std::endl;

  session->Close();
  return 0;
}
