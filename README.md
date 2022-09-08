# レイトレ合宿8 レンダラー & デノイザー？
[レイトレ合宿8](https://sites.google.com/view/raytracingcamp8/)用に書いたレンダラーとデノイザーのコード置き場。
合宿用に書いたコードなので人に見せるようなコードではないですが、概要：
- レンダラー：Neural Radiance Cachingをボリューメトリックパストレーサーに実装、多重散乱ボリュームを高速にレンダリングできるようにしました。オフラインレンダリング用途のため、オリジナルのNRCとはいくらか実装を変えてあります。よりオリジナルに忠実な実装は[GfxExp](https://github.com/shocker-0x15/GfxExp)で公開しているので良かったらそちらをご参照ください。
- デノイザー：サブ部門用。デノイザーと呼ぶのがおこがましいトンデモコード。超適当なMLPに周辺ピクセルのフィーチャーを入力して輝度を推定するというもの。全くうまくいかなかった。

![VLR](output_top.png)\
Cloud is composed of a cloud from "[Walt Disney Animation Studios Cloud Data Set](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/)" and Stanford bunny [from McGuire CG Archive](https://casual-effects.com/data/) converted to a volume using Houdini.\
The license of this rendered image follows the [Creative Commons Attribution-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/).