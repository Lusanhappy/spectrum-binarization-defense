# Installation

安装依赖：
1. 安装pytorch
   ```
   $ conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch
   ```
2. 安装librosa
   ```
   $ conda install -c conda-forge librosa
   ```
3. 安装praat
   ```
   $ pip install praat-parselmouth
   ```

# Usage

## Datasets

数据集下载：

- 百度云：
  链接：https://pan.baidu.com/s/1jfDq-8RZ_PM1YOupDtRJwQ 
  提取码：a7zm

- 天翼云：
  https://cloud.189.cn/t/yAriEzmim2m2 (访问码:m9as)


解压数据集*datasets.7z*，将*datasets*文件夹与脚本放在同一工作目录下（数据集大小大概有23G）。

如果需要自行指定训练数据集、测试数据集，请修改*default_config.py*文件。

## Training

- 训练模型M1（3CNN_2DNN，使用log-mel作为特征）
  ```
  $ python train.py --epoch 40 --model_name cnn3_dnn2 --use_melspec 1 --use_formants 0 --use_pitch 0
  ```
- 训练模型M2（3CNN_2DNN，使用log-mel、formants、pitch作为特征）
  ```
  $ python train.py --epoch 40 --model_name cnn3_dnn2 --use_melspec 1 --use_formants 1 --use_pitch 1
  ```
- 训练模型M3（CNN_Attention）
  ```
  $ python train.py --epoch 40 --model_name cnn_attention
  ```

或者修改*default_config.py*文件，可以调整其他参数，然后直接运行*train.py*.

## Test

- 测试模型M1（3CNN_2DNN，使用log-mel作为特征）
  ```
  $ python test.py weights/cnn3_dnn2_m1.pth --model_name cnn3_dnn2 --use_melspec 1 --use_formants 0 --use_pitch 0
  ```
- 测试模型M2（3CNN_2DNN，使用log-mel、formants、pitch作为特征）
  ```
  $ python test.py weights/cnn3_dnn2_m2.pth --model_name cnn3_dnn2 --use_melspec 1 --use_formants 1 --use_pitch 1
  ```
- 测试模型M3（CNN_Attention）
  ```
  $ python test.py weights/cnn_attention_m3.pth --model_name cnn_attention
  ```
- 对指定目录的wav音频文件测试，使用*adv_dirs*和*cln_dirs*参数，例如：
  ```
  $ python test.py weights/cnn3_dnn2_m2.pth --adv_dirs path1 path2 --cln_dirs path3 path4
  ```

或者修改*default_config.py*文件，可以调整其他参数，然后直接运行*test.py*.
