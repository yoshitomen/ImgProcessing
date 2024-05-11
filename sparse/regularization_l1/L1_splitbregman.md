
スパースモデリングによるL1ノルム最適化の際に最小化する式は
$$F=\frac{1}{2}|I_{out}-I_{in}|^2+\lambda |I_{out}|_1$$
である. $\lambda$は正則化パラメータで基準値(後の$\mathrm{Thr}$)へ近づける強度係数となっている. 第一項で入力-出力間の差分, 第二項がL1ノルムで0に近い解を出しやすくする効果がある. 基準を0ではなく$Thr$にしたい時は
$$\begin{equation}
F=\frac{1}{2}|I_{out}-I_{in}|^2+\lambda |I_{out}-\mathrm{Thr}|_1
\end{equation}$$
のように第二項を変形する. 例えば白色の画素をbackgroundにしたい場合は$\mathrm{Thr}=255$となる.

画像のノイズ除去を行う場合, i番目の画素を
$$
\begin{aligned}
a_ie_i&=I_{out}-\mathrm{Thr}\ (\ a_i\ge0, |e_i|=1)\\b_i&=I_{in}-\mathrm{Thr}
\end{aligned}
$$
のように表すことにする. このとき, $a_i$を最適化することが$I_{out}$の最適化になることが分かる. これを用いて(1)式を変形すると
$$
F=\sum_i\frac{1}{2}(a_ie_i-b_i)^2+\lambda\sum_i a_i
$$
となる. $F$を最小値にする場合, $e_i=\mathrm{sgn}(b_i)$となることと
$$
(a_ie_i-b_i)^2=(\mathrm{sgn}(b_i)\times(a_i-|b_i|))^2=(a_i-|b_i|)^2
$$
であることに注意して計算すると
$$\begin{aligned}
F&=\frac{1}{2}\sum_i(((a_i-|b_i|)+\lambda)^2 -2(a_i-|b_i|)\lambda-\lambda^2)+\lambda\sum_i a_i\\
&=\frac{1}{2}\sum_i(a_i-t_i)^2-\lambda\sum_i(a_i-|b_i|)-\frac{1}{2}\sum_i\lambda^2+\lambda\sum_i a_i\\
&=\frac{1}{2}\sum_i(a_i-t_i)^2+\lambda\sum_i t_i+\frac{1}{2}\sum_i\lambda^2
\\&(\because t_i=|b_i|-\lambda, a_i\geq0)
\end{aligned}
$$
となる. ただし画像のノイズ除去などスパース化したい物理量が輝度値の場合, $\mathrm{Thr}$が正の値になることを暗に仮定しているため負の値を含む場合は陽に計算できないことに注意する.

以上より
$$
\begin{equation}
F=\frac{1}{2}\sum_i(a_i-t_i)^2+\mathrm{const}
\end{equation}
$$
となり, $F$を最小化する解は
$$
a_i = \begin{cases}
    t_i & (t_i \ge 0) \\
    0 & (t_i < 0)
  \end{cases}
$$
となり,
$$
I_{out, i} = \begin{cases}
    (|b_i|-\lambda)\mathrm{sgn}(b_i)+\mathrm{Thr}& (|b_i| \ge \lambda) \\
    \mathrm{Thr} & (b_i < \lambda)
  \end{cases}
$$
がスパースモデリングによる再構成結果となる.