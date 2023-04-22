/*#include "./MNIST.h"

void testMNIST() {
   MNIST::UByteTensorFile images(C_TEST_DATA_DIR"/MNIST/train-images.idx3-ubyte");
   MNIST::UByteTensorFile labels(C_TEST_DATA_DIR"/MNIST/train-labels.idx1-ubyte");

   NeuralNetwork network;

   network.addLayer(Size(2, 28, 28));
   //network.addLayer(Size(1, 28 * 28));
   //network.addLayer(Size(1, 28 * 28 / 2));
   //network.addLayer(Size(1, 28 * 28 / 4));
   network.addLayer(Size(1, 10));

   if (0) {
      Scalar learningRate = 0.1;
      int count = images.datacount;
      int index = 1;
      while (index < count) {
         for (int k = 0; k < 100 && index < count; k++) {
            Tensor2D* inData = images.getTensor2D(index);
            Tensor1D* outData = labels.getOrdinalTensor1D(index, 10);
            network.feed(inData);
            network._learn(outData, 0.01);
            network._mutate(learningRate);
            index++;
         }
         Tensor2D* inData = images.getTensor2D(0);
         Tensor1D* outData = labels.getOrdinalTensor1D(0, 10);
         network.feed(inData);
         network._learn(outData, 0.01);
         network.printResults();
      }
   }
   else {
      Scalar learningRate = 0.1;
      Tensor2D* inData = images.getTensor2D(1);
      Tensor1D* outData = labels.getOrdinalTensor1D(0, 10);
      saveBitmap(C_TEST_OUTPUT_DIR"/test.bmp", inData);
      for (int k = 0; k < 1000; k++) {
         network.check();
         network.feed(inData);
         network.feedback(outData);
         network.learn(learningRate);
         network.printResults();
         printf("> error: %lg\n", network.layers.back()->error());
      }
   }
}

struct Plane {
   std::vector<Scalar> direction;
   Scalar offset = 0.0;

   Scalar eval(std::vector<NeuronState*>& inputs) {
      Scalar acc = this->offset;
      for (int i = 0; i < this->direction.size(); i++) {
         acc += this->direction[i] * inputs[i]->computed;
      }
      return acc;
   }
   Scalar magnitude() {
      Scalar acc = 0.0;
      for (auto w : this->direction) acc += w * w;
      return sqrt(acc);
   }
   void setNormalizedOffset(Scalar normOffset) {
      this->offset = normOffset * this->magnitude();
   }
};

void testNeuron() {
   Plane plane;
   Scalar factor = 10.0;
   plane.direction.push_back(factor * 1.0);
   plane.direction.push_back(factor * 1.0);
   plane.setNormalizedOffset(-0.7);

   std::vector<NeuronState*> inputs;
   inputs.push_back(new NeuronState(NeuronId(1, 0)));
   inputs.push_back(new NeuronState(NeuronId(2, 0)));

   Neuron* output = new Neuron(NeuronId(1, 1));
   output->connect(*inputs[0]);
   output->connect(*inputs[1]);
   output->setup();

   Scalar learnRate = 0.01;

   int count = 1000;
   int displayFreq = count / 20;
   for (int i = 0; i < count; i++) {

      // Compute random entry
      inputs[0]->setInnerValue(randomScalar(0.0, 1.0));
      inputs[1]->setInnerValue(randomScalar(0.0, 1.0));
      output->compute();
      output->expected = plane.eval(inputs);

      // Check result
      if (i % displayFreq == 0) {
         printf("--------------------------------------\n");
         printf("  > error: %.3lg\t[%.3lg]\n", Scalar(output->computed - output->expected), Scalar(output->computed));
#if 1
         for (auto& link : output->links) {
            printf("  > link:\n");
            printf("       covar=%.3lg\n", link.covariance);
            printf("       coexcitation=%.3lg\n", link.coexcitation);
            printf("       coinhibition=%.3lg\n", link.coinhibition);
            printf("       weight=%.3lg\n", link.weight);
         }
#endif
      }

      output->backpropagateImpulse(1);
      for (auto x : inputs) x->learnMean(learnRate);
      output->learn(learnRate);
      output->mutate(0.01);
   }
}

void main() {
   srand(10);
   testNeuron();
   //testMNIST();
}
*/