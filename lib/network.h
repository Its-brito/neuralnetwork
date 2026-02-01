#pragma once

#include <vector>
#include <cmath>
#include <functional> // Required for std::function
#include <numeric>    // Required for std::inner_product


namespace NN {
  class Node {
    public:
      // Public API
      float output=0.0f;
      char act;
      std::vector<float> input;
      std::vector<float> weights; // from previous nodes
      float bias=0.f;

      void forward() {
        if (input.size() != weights.size()) {
          // Error handling: vectors must match size
          return;
        }
        calculateOutput();
      }
    private:
      float activation(char name,float x){
        if(name=='s'){return 1/(1+exp(x));}
        if(name=='l'){return std::max(0.1f*x,x);}
        if(name=='r'){return std::max(0.0f,x);}
        if(name=='t'){return tanh(x);}
        return 1;
      }
    // Simple calculation: each node output is equal to \sigma sum weights*input
      float calculateOutput(){
        float sum=0;
        for(size_t i=0 ; i<input.size() ; i++){
          sum+=weights[i]*input[i];
        }
        sum+=bias;
        return output=activation(act,sum);
      }
  };

  class Layer{
  public:
    int width; //number of nodes in layer
    int depth; // number of layers
    Node node;

  private:
    void connectLayers(){
      for(int i=1;i<depth,i++){
        for(int j=0;j<width;i++){

        }
      }
    }


  };
}
