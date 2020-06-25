#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc,const char*argv[]){

  torch::jit::script::Module module;
  try {
      module = torch::jit::load("/home/nas/user/kbh/End-to-End-VAD/traced/tracedState.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  // 입력값 벡터를 생성합니다.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 15,3, 224, 224}));
  // TODO hidden layer 의 초깃값을 넣어줘야한다.

  //모델을 실행한 뒤 리턴값을 텐서로 변환합니다.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  return 0;
}
