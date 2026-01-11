
### 模型测试

测试对抗样本：
```shell
python test.py weights/logs-w2v-dt-2023-02-03-19-03-41/epoch-30_2023-02-03-19-08-10_trnacc-0.9822_valacc-0.9878.pth --adv_dirs samples/noise_boost
```

测试正常样本：
```shell
python test.py weights/logs-w2v-dt-2023-02-03-19-03-41/epoch-30_2023-02-03-19-08-10_trnacc-0.9822_valacc-0.9878.pth --cln_dirs samples/musics
```

也可以组合使用，详情参考*test.py*文件。

### 生成降采样样本


```shell
python perturb.py --wav-dir samples/noise_boost --type downsample --save-dir samples/noise_boost_perturbed/downsample-5600 --target-sr 5600
```

### 生成噪声样本


```shell
python perturb.py --wav-dir samples/noise_boost --type noise --save-dir samples/noise_boost_perturbed/noise-100 --noise-level-int16 100
```

