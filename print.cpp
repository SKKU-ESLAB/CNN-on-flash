#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ios>

int main(int argc, char *argv[]){
			  if (argc != 4) {
								printf("usage: ./print <matrix> <rows> <cols>\n");
								return 0;
				}
				int rows = atoi(argv[2]);
        int cols = atoi(argv[3]);
				float* buf_1 = (float*) malloc(rows*cols*sizeof(float));
				std::string filename_1 = std::string(argv[1]);
				std::ifstream result_1(filename_1,std::ios::binary);
				result_1.read((char*)buf_1, rows*cols*sizeof(float));
				
				printf("\n");
				for (int i=0; i<rows; i++) {
						for (int j=0; j<cols; j++){
                printf("%d ", ((int)(float) buf_1[rows*i+j]));
					  }
						printf("\n");
				}
				return 0;
}
