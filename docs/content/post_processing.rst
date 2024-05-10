Post-processing and plotting
============================

A post processing script is also available to plot all results. To run it type

```sh
python -m linfa.plot_res -n NAME -i IT -f FOLDER
```

where `NAME` and `IT` are again the experiment name and iteration number corresponding to the result file of interest, while `FOLDER` is the name of the folder with the results of the inference task are kept. Also the file format can be specified throught the `-p` option (options: `pdf`, `png`, `jpg`) and images with dark background can be generated using the `-d` flag. 
