#pragma once
#include "tensor.h"
#include <fstream>

namespace MNIST {

struct uint32_HE_t {
   uint32_t value;
   operator uint32_t() {
      return _byteswap_ulong(value);
   }
};

   struct UByteTensorFile {

      struct tHeader {
         uint8_t magic[3];
         uint8_t dim;
         uint32_HE_t count; // count defined by dim
         uint32_HE_t sizes[4]; // count defined by dim
      };

      uint8_t dim = 0;
      uint32_t sizes[4] = {};
      uint32_t datacount = 0;
      uint32_t datasize = 0;
      uint8_t* datas = 0;

      UByteTensorFile(std::string path) {
         std::ifstream f(path, std::ios_base::in | std::ios_base::binary);
         checkAssert(f.is_open());

         tHeader header;
         f.read((char*)&header, sizeof(tHeader));
         checkAssert(header.dim <= 4);
         this->dim = header.dim - 1;
         this->datacount = header.count;
         this->datasize = 1;
         for (int i = 0; i < this->dim; i++) {
            this->sizes[i] = header.sizes[i];
            this->datasize *= this->sizes[i];
         }

         int bufsz = this->datacount * this->datasize;
         f.seekg(sizeof(uint32_t) * (2 + this->dim), std::ios_base::beg);
         this->datas = (uint8_t*)malloc(bufsz);
         f.read((char*)this->datas, bufsz);
      }
      Tensor1D* getOrdinalTensor1D(int index, int range) {
         if (dim != 0) return 0;
         if (index >= this->datacount) return 0;
         uint8_t ordinal = this->datas[index * this->datasize];
         auto data = new Tensor1D(range);
         data->values[ordinal] = 1.0;
         return data;
      }
      Tensor1D* getTensor1D(int index) {
         if (dim != 1) return 0;
         if (index >= this->datacount) return 0;
         auto data = new Tensor1D(sizes[0]);
         data->fill(&this->datas[index * this->datasize], this->datasize);
         return data;
      }
      Tensor2D* getTensor2D(int index) {
         if (dim != 2) return 0;
         if (index >= this->datacount) return 0;
         auto data = new Tensor2D(sizes[0], sizes[1]);
         data->fill(&this->datas[index * this->datasize], this->datasize);
         return data;
      }
   };
}
