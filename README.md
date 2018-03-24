# DeepLearningでアップサンプリングする
オーディオ界隈はオカルトっぽく見えていたので、今までどうしようと思っていたのですが、簡単な感じで結果がでました  

世の中、音のアップサンプリングや音質がよくなるような細工に本当に余念がないのですが、ディープラーニングでも簡単に対応することは可能です。  

世の常としてA/D変換されたデータは元のデータが欠落するから、音の復元は無理だと言われ[3]てきましたが、機械学習を使えばその制限は突破できます。  


## High Resolution
ハイレゾは96kHz/24bitという高いサンプリング数と、高い解像度を誇ります。  

通常、YouTubeでは44kHz/16bitで音楽が再生されるので、及ばないのですが、22kHz/16bitの音源を44kHz/16bitに引き上げてみます。  

<p align="center">
  <img width="450px" src="https://user-images.githubusercontent.com/4949982/37856908-79a2b6de-2f3a-11e8-9ee7-885d2492a313.png">
</p>
<div align="center"> 図1. 今回やりたいこと </div>

この中間を補填するロジックに深層学習を組み込みます。  

## 今回使用したネットワーク
幾つかやり方にはコツがあって、実は音の16bit値をfloatに変換したのでは、うまく行きません。これは、DNNがあまりにも小さい値には反応しないし、この小さなさが、音の善し悪しを分けたりするからです。    

そのため、16bitをバイナリ（２進数）表記に変換して、音のある波形の前後25サンプルを切り出します。  

25サンプルから、44kHzで本来存在していただろう、音の波形を補完します。  

人の声や楽器など、ある程度のレンジを取って様子を換算して音の並ごとに特性があるとの仮説があるからです。  

ネットワークでは、双方向のLSTMのネットワークを用いました。  

<p align="center">
  <img width="650px" src="https://user-images.githubusercontent.com/4949982/37856898-6adc7950-2f3a-11e8-823e-5090eb81da1e.png">
</p>
<div align="center"> 図2. 全体のデータの流れ　</div>

コードで書くと、こんな感じです。(Bi=Bidirection, TD=TimeDistribute)
```python
input_tensor1 = Input(shape=(50, 16))
x           = Bi(CuDNNLSTM(300, return_sequences=True))(input_tensor1)
x           = TD(Dense(500, activation='relu'))(x)
x           = Bi(CuDNNLSTM(300, return_sequences=True))(x)
x           = TD(Dense(500, activation='relu'))(x)
x           = TD(Dense(20, activation='relu'))(x)
decoded     = Dense(1, activation='linear')(x)
print(decoded.shape)
model       = Model(input_tensor1, decoded)
model.compile(RMSprop(lr=0.0001, decay=0.03), loss='mae')
```

## 実験
ボーカロイドの曲である[「wave」をreworuさんが歌ったもの](https://www.youtube.com/watch?v=36SxEHQeDi8)を利用しました  

YouTubeからwaveファイルを取り出すことはできるので、ダウンロードしたら、入力用に22kHzにダウンサンプルします。(著作権の関係で音源は添付しませんのでご自身でご用意してください)  

[https://github.com/GINK03/keras-rnn-music-super-resolution]

```console
$ python3 10-scan.py
```
オリジナルの曲と、ダウンサンプルした曲のペアのデータ・セットを作成してnumpyに変換します  
```console
$ python3 20-make-dataset.py
```
**学習**  

GTX1080Tiで二時間程度です  
(音楽の前８割を学習に使い、残り２割を評価に回します)
```console
$ python3 rnn-super-resolution.py --train
```

**アップサンプリング**  
```console
$ python3 rnn-super-resolution.py --predict
```

## 結果（誤差評価）
ぶっちゃけ、聞いても定性的な評価になってしまうというのが本音なので、テストデータにおけるMean Absolute Error(平均絶対誤差)をみて良くなっていることを確認します。  

```console
$ python3 eval.py              
オリジナルデータ                            0.0
22kHzデータ                               1260.0746398721526
ディープラーニングでアップサンプリングしたデータ 610.2184827578526
```
値が少ないほうがいいのですが、たしかに、22kHzのデータそのものより音質は改善していることがわかりました。  


## 結果（聞いてみる）

**オリジナル44khz** .

https://soundcloud.com/sgemuj01eczp/origin

**ダウンサンプル22kHz**  

https://soundcloud.com/sgemuj01eczp/degradation-1

**機械学習でアップサンプリング22khz->44khz**   

https://soundcloud.com/sgemuj01eczp/yp-orig-5

よく聞き分けると、ノイズのような音源が、ところどころ機械学習では混じっていることがわかるかと思います(課題)  

## まとめ
ネットワークの大きさや、出力を工夫することで、44khz -> 88khzも可能だと思うし、単純なフィルタを超えたアップサンプリングが可能だと思います。  

オーディオ沼は怖いのでほどほどにしておきたいですが、簡単にできるので、やってみる価値はあるかもしれません。  

最後に実は全然別で、audio super resolutionというものをやってらっしゃる方を発見して、これは、音を画像のように捉えてアップサンプリングするようです[2]。  


## 参考文献＆オカルト
- [1] [mp3音源を“アップサンプリング”で高音質化できるか試してみた](https://kakakumag.com/pc-smartphone/?id=9459) 
（データフォーマットを変形しただけでアップサンプリングできていないように見える）  

- [2] [AUDIO SUPER-RESOLUTION USING NEURAL NETS](https://arxiv.org/pdf/1708.00853.pdf)

- [3] [再びアップサンプリングの功罪について](http://flac.aki.gs/bony/?p=3683)
