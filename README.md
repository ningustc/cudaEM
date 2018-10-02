## Welcome to GitHub Pages

cudaEM is a cross platfrom Multi-GPU based package used for multislice simulation. This package eliminates the bandwidth limit of GPU RAM with CUFFTCallback, and makes full use the computation power of CUDA Cores. 
This package has no other dependance except for CUFFT library. 
In linux and Mac OS, the CUFFT callback functions are available after statically linking to cufft-static library.
In Windows OS, the CUFFT callback functions are disabled due to the lack of static libraries under Windows platfrom.


![alt text](https://github.com/ningustc/cudaEM/blob/master/LAADF.bmp)
### Markdown

```markdown
Installation on Linux and Mac
cd build
make all
./cudaEM
Then you can run compiled cudaEM 
```

```markdown
Installation on windows
please install CMake GUI version and add the CMakeList.text. 
Do remember add the cuda directory into your environmental variables.

Then you can run compiled cudaEM with Visual Studio or QT
```


### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ningustc/cudaEM/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

msens@nus.edu.sg

