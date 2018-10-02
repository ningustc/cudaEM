## Welcome to GitHub Pages

cudaEM is a cross platfrom Multi-GPU based package used for multislice simulation. This package eliminates the bandwidth limit of GPU RAM with CUFFTCallback, and makes full use the computation power of CUDA Cores. 
This package has no other dependance except for CUFFT library. 
In linux and Mac OS, the CUFFT callback functions are available after statically linking to cufft-static library.
In Windows OS, the CUFFT callback functions are disabled due to the lack of static libraries under Windows platfrom.

### Markdown

```markdown
Installation on Linux and Mac
cd build
make all
./cudaEM
Then you can run compiled cudaEM 
[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ningustc/cudaEM/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
