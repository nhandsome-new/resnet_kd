# Knowledge Distillation

## Concept
- Base Line : ResNet18 
- Offline KD : Teacher(ResNet50)
    - [teacher model](https://github.com/huyvnphan/PyTorch_CIFAR10)
- Online KD : Teatcher(ResNet50)
    - 
- Self KD : ResBet18

- Dataset : CIFAR10

- With subset augmentation
- W/O subset augmentation


### 'unexpected key "module.xxx.weight" in state_dict'
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])

### middle_feature = middle_output

- 27 : only student : 74.4 (2022-08-16_13-44-03)
- 28 : self : 77.89 (2022-08-16_15-00-11)
- 29 : self teacher : 77.95(2022-08-16_16-24-05)
- 30 : teacher : 77.56 (2022-08-16_18-09-44)
- 31 : teacher without mixup : 76.86 (2022-08-17_02-50-18)
- 32 : self teacher with big


# 知識蒸留(Knowledge Distillation)を使ってResNet18をより賢くしてみよう

こんにちは、機械学習チームのハンです。

知識蒸留(KD:Knowledge Distillation)というものが気になり、簡単なResNetモデルで色々実験を行ってみたので紹介したいと思います。

## Knowledge Distillationとは？
「あるモデルが学んだ知識を他のモデルに移す」という概念であり、[Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/pdf/1503.02531.pdf)で最初提案されたものです。

最近の機械学習モデルはその精度は段々上がっていますが、モデルが巨大化されていて、実際にデプロイしたり個人が利用したりするには負担になるものも多いです。下の図で確認できるように、Teacherモデルの知識を蒸留しStudentモデルに移して「より軽くて良い精度を持つモデルを作る」のが、KDの目的だと考えられます。

- Image : KD_concept

## どのような知識を利用するか？
KDで利用される知識(Knowledge)は大きく下記の3つに分けられます。
- Response-based Knowledge
- Feature-based Knowledge
- Relation-based Knowledge

### Response-based Knowledge
モデルのResponse、つまり「モデルのOutput」を知識として定義する方法です。巨大なTeacherモデルのOutput（Softmax・Logit）には、豊かな情報が入っているという考えで、Outputの分布を知識として扱います。
- Image : Response-based
下図はResponse-base Knowledgeを用いた一般的なKDモデルの構成ですが、右上のDistillation Lossの部分に注目してください。Pre-trainされた「TeacherモデルのOutputを、真似する知識として学習」していることがわかります。

### Feature-based Knowledge
モデルの中間層のFeature-Map、「中間層のOutput」を知識として扱う方法で、この特徴は「Hint」とも呼ばれます。下の図はFeature-based Knowledgeを用いたKDモデルの構造を表していますが、Studentモデルは「Teacherモデルの中間層から知識を得て、同様な中間層Outputを作れる」ように学習を進めます。

この方法は上記したResponse-based Knowledgeと比べ、細くて深い層を持つネットワークの学習に役に立つと思われています。しかし、中間層のOutputの形はモデル・深さによって様々であり、Lossを実装する時に手間がかかる場合もあります。
- Image : Feature-based 

### Relation-based Knowledge
モデル中の「レイヤーの間に存在する関係」を知識として扱う方法です。上記２つの方法では、Teacher・Studentモデルのある出力を直接１：１マッチングして比較しますが、Relation-based Knowledgeでは、個別モデルの中間層Hint間の関係を求め、その関係を比較します。例えば、中間層出力間の距離を関係として扱い、「Teacherモデルと似たような関係をStudentが学習する」ような形になります。
- Image : Relation-based


## 知識をどのように移動させるか？
KDを行う方式として、以下の３つのやり方があげられます。
- Offline Distillation
- Online Distillation
- Self Distillation
- Image

### Offline Distillation
「Pre-trainしたTeacherモデルの知識をStudentモデルに学習させる」方法で、一番簡単で一般的なKD方法になります。KDプロセス上、TeacherモデルはFreezeされ、「入力に対して知識になるOutputを推論する」役割を行い、Studentモデルは「Teacherモデルの知識を用いて学習」を行います。一般にTeacherは「スケールが大きく良い精度を持つモデル」が考えられ、個人的には「大きいモデルの知識を蒸留し軽いモデルに移す」というKDの目的にぴったりの方法だと思います。
特徴としては、Pre-trainされたTeacherモデルが要るので「２Stage方式」になることです。

### Online Distillation
KDプロセス上、「Teacher・Studentモデルがお互いに学習を行い、知識を共有する方法」です。上の図は、スケールが大きくより学習能力のあるTeacherモデルの知識をStudentモデルに蒸留することを表しています。
しかし、Online Distillationの応用としては、Teacherモデル無しで
- 「似たスケール・複数モデル」をStudentとして使用
- 「同じネットワーク・設定の違うモデル」をStudentとして使用
する方法もあります。([Deep Mutual Learning](https://arxiv.org/pdf/1706.00384.pdf))
Offline Distillationと異なり、End-to-End学習ができる・モデルお互いに知識の共有ができるという特徴があります。

### Self Distillation
「一つのモデルでKDを行う」方法で、ネットワーク中でTeacherになるある特徴を選択し、その特徴をStudentが学習します。例えば、
- 「同じイメージ・異なるAugmentation結果」を一つのモデルに通し、お互いのOutputをTeacher・Studentにする
- 「最終レイヤー・中間レイヤー」のOutputをTeacher・Studentにする
などの方法が挙げられます。
一つのモデルを用いているので上記２つの方法と比べ、学習時間・パラメータ数の面でメリットになる特徴があります。

### ３つの方法と一般的なモデル学習の比較

|基準|比較|
|:---:|:---:|
|パラメータ数|Online >= Offline > Self >= Normal|
|学習時間|Offline > Online > Self > Normal|
|精度|Offline ・ Online ・ Self > Normal|

上記のテーブルは「同じStudentモデルを学習する」という前提で考えてみたモデルの比較です。（３つのKD方法＋Normal学習）
学習パラメータ・学習時間を考えると、Self Distillationの方が選ばれると思いますが、「既にPre-trainされたTeacherモデルが存在する」「各KD方法には細かいやり方が存在する」「条件によって精度は変わってくる」という面を考えてみると、どっちの方法が優れているとは言い難いと思います。


## 今回試してみたことは？
ResNetを用いて、CIFAR100データの分類モデルを学習してみました。KDを使ってない普通なモデル(Baseline)を学習し、様々な知識蒸留方法を試しながら精度の変化を確認してみます。

### Baselineモデル
BaselineになるResNetモデルは下記の設定で学習されました。
- ResNetバージョン：18
- クラス数：100
- Loss関数：Cross Entropy Loss
- Optimizer：SGD
- Scheduler：Cosine Annealing with Warm Up
- Epochs : 200

#### Cosine Annealing with Warm Up
- Image


### Offline Distillation モデル
ResNet18をStudentモデルで、下のようなOffline Distillationモデルを構成しました。


### Self Distillation モデル

### Offline + Self Distillation モデル




## 参考資料
([Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525.pdf))