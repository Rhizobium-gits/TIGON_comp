# TIGON_comp
https://github.com/yutongo/TIGON

ここからデータをダウンロードしてEMT_dataのzipを解凍してからDriveにアップロードしてからCoLab上でマウント

numpyのnp.dtypesの部分をnp.dtypeに書き換える必要がある(バージョンの都合)

<h2>手順</h2>
<h3>1.Driveに接続</h3>
<h3>2.環境構築</h3>

pytorch 1.13.1  # pytorchのpyは消す

Scipy 1.10.1 # Visualizationで使う

TorchDiffEqPack 1.0.1

torchdiffeq 0.2.3

numpy 1.23.5  # AE, Visualizationで使う

seaborn 0.12.2  

matplotlib 3.5.3  # AEで使う

scanpy 1.9.3  # AEで使う

utility  # Visualizationで使う

<h3>3.AEの読み込み(AE.ipynb)</h3>

https://github.com/yutongo/TIGON/blob/main/Notebooks/AE.ipynb

<h3>4.Trainingの読み込み(Training.ipynb)</h3>

https://github.com/yutongo/TIGON/blob/main/Notebooks/Training.ipynb

<h3>5.Visualizationの読み込み(Visualization.ipynb)</h3>

https://github.com/yutongo/TIGON/blob/main/Notebooks/Visualization.ipynb

・Generate the plot of regulatory matrix(plot_jac_v)

・Generate the plot of gradient of growth(plot_grad_g)

・Generate the plot of trajecotry(plot_3d)
