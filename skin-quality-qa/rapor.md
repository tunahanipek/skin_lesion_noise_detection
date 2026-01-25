# Derin Öğrenme Kullanarak Dermoskopik Görüntülerde Gürültü Sınıflandırması: Kapsamlı Güncelleme ve Stratejik Yol Haritası

## Yönetici Özeti

Dermatoloji alanında Bilgisayar Destekli Tanı (CAD) sistemlerinin güvenilirliği, teşhis algoritmalarını besleyen veri kalitesine kopmaz bir bağla bağlıdır. "Noise Classification in Dermoscopic Images Using Deep Learning" başlıklı temel çalışma, dermoskopik görüntülerdeki kıl, hava kabarcıkları ve hareket bulanıklığı gibi klinik artefaktları (yapay bozulmalar), teşhis açısından kritik lezyon özelliklerinden ayırt edebilen ön işleme hatlarına duyulan ihtiyacı vurgulamıştır.<sup>1</sup> Orijinal raporun yayınlanmasından bu yana, tıbbi görüntü analizi alanı, Vizyon Transformer'ları (ViT), EfficientNetV2 gibi gelişmiş Evrişimli Sinir Ağları (CNN) ve sofistike çok etiketli (multi-label) öğrenme çerçevelerinin ortaya çıkmasıyla köklü bir paradigma değişimi geçirmiştir. Bu kapsamlı araştırma raporu, orijinal çalışmanın 2017-2025 yılları arasındaki literatür bulgularıyla entegre edilerek yapılan kapsamlı bir güncellemesidir. Bu doküman, metodolojik boşlukları belirler, mimari iyileştirmeler önerir ve dermoskopik artefakt taksonomisini yeniden tanımlar. 50'den fazla son teknoloji çalışmanın verilerini sentezleyerek, başlangıçtaki MobileNetV2 tabanlı çerçeveyi, gerçek dünya dermatolojik verilerinin karmaşıklığını yönetebilen sağlam, klinik olarak uygulanabilir bir kalite değerlendirme sistemine yükseltmek için ayrıntılı bir yol haritası sunar.

## 1\. Giriş: Dermoskopide Veri Kalitesinin ve Gürültü Yönetiminin Önemi

Dermoskopik görüntüleme, cilt yüzeyinin altındaki morfolojik yapıların görselleştirilmesini sağlayan ve melanom gibi malign (kötü huylu) cilt tümörlerinin erken teşhisinde devrim yaratan non-invaziv bir tekniktir. Bu teknik, çıplak gözle yapılan muayeneye kıyasla tanısal doğruluğu önemli ölçüde artırsa da, elde edilen görüntülerin kalitesi, görüntüleme sürecinde ortaya çıkan çeşitli artefaktlar nedeniyle sıklıkla bozulmaktadır. Orijinal rapor, özellikle Uluslararası Cilt Görüntüleme İşbirliği (ISIC) arşivi gibi geniş ve heterojen kaynaklardan derlenen veri setlerinin, otomatik tanı algoritmalarının performansını ciddi şekilde düşüren gürültülerle dolu olduğunu doğru bir şekilde tespit etmiştir.<sup>1</sup> Görüntü kalitesindeki bu varyasyonlar, yapay zeka modellerinin eğitiminde kullanılan verilerin tutarlılığını zedelemekte ve modelin genelleme yeteneğini kısıtlamaktadır.

### 1.1 Tıbbi Yapay Zekada "Çöp Girer, Çöp Çıkar" Fenomeni ve Ötesi

Orijinal çalışmanın temel önermesi olan "gürültü sınıflandırmasının kritik bir ön işleme adımı olduğu" gerçeği, günümüzde geçerliliğini korumakla kalmamış, aciliyeti daha da artmıştır. Ancak sorun sadece "gürültü" değildir; literatürdeki son gelişmeler, artefaktların pasif birer kirlilik olmaktan öte, derin öğrenme modellerinin sınıflandırma görevlerinde hile yapmak için kullandığı "karıştırıcılar" (confounders) veya "kestirme yollar" (shortcuts) olarak işlev gördüğünü ortaya koymaktadır.

Örneğin, biyopsi ile doğrulanmış melanom vakalarında cerrahi mürekkep işaretlerinin veya cetvel görüntülerinin daha sık bulunması nedeniyle, sınıflandırıcıların bu nesneleri malignite (kötü huyluluk) ile ilişkilendirmeyi öğrendiği kanıtlanmıştır.<sup>2</sup> Bu durum, "Clever Hans" etkisi olarak bilinir; model, lezyonun biyolojik özelliklerini öğrenmek yerine, görüntüdeki yapay işaretleri takip ederek yüksek doğruluk skorları elde eder, ancak klinik pratikte felaketle sonuçlanabilecek hatalar yapar.<sup>4</sup> Benign (iyi huylu) bir nevüsün, sırf üzerinde bir cerrahi kalem izi olduğu için melanom olarak sınıflandırılması, veri kalitesi kontrolünün sadece bir teknik detay değil, bir hasta güvenliği meselesi olduğunu göstermektedir. Bu nedenle, gürültü sınıflandırma sistemi, sadece artefaktları tespit etmekle kalmamalı, aynı zamanda bu sahte korelasyonları ortaya çıkaracak ve engelleyecek yetenekte olmalıdır.

### 1.2 Orijinal Kapsamın Eleştirisi ve Modernizasyon İhtiyacı

Referans alınan orijinal çalışma, MobileNetV2 mimarisini kullanarak görüntüleri sekiz ayrı gürültü türüne (Kıllı, Bulanık, Kabarcıklı, Metin/Lisans, Temiz, İşaretli, Vinyet ve Kalem) ayırmayı önermiştir.<sup>1</sup> Bu yaklaşım, gürültü türlerini granüler bir şekilde ele alması bakımından yenilikçi olsa da, her görüntünün sadece ve sadece bir kategoriye ait olduğunu varsayan "tek etiketli" (single-label) sınıflandırma çerçevesine dayanmaktaydı.

Oysa güncel klinik araştırmalar ve veri analizleri, bu varsayımın hatalı olduğunu göstermektedir. Tek bir dermoskopik görüntüde birden fazla artefaktın eş zamanlı olarak bulunması kuraldışı değil, olağan bir durumdur. Örneğin, bir lezyon görüntüsü aynı anda hem yoğun kıl örtüsüyle kaplı olabilir, hem jelleşme hatasından kaynaklı hava kabarcıkları içerebilir, hem de odaklama hatasından dolayı bulanık olabilir.<sup>2</sup> Orijinal raporun önerdiği model, bu durumda görüntüyü tek bir sınıfa zorlayarak (örneğin sadece "Kıllı" diyerek) diğer gürültü kaynaklarını ihmal etmekte ve veri temizleme sürecini eksik bırakmaktadır. Bu raporun güncellenmesindeki en temel gerekliliklerden biri, sistemin **Çok Etiketli (Multi-Label)** öğrenme paradigmasına geçirilmesidir.

## 2\. Dermoskopik Görüntü Analizinin Evrimi: Kapsamlı Literatür Taraması (2009-2025)

Gerekli güncellemeleri bağlamına oturtmak için, dermoskopik görüntü analizinde artefakt yönetimi ve sınıflandırma tekniklerinin tarihsel gelişimini, erken dönem görüntü işleme tekniklerinden günümüzün multimodal (çok modlu) yapay zeka sistemlerine kadar izlemek gerekmektedir.

### 2.1 El Yapımı Öznitelikler ve Morfolojik İşlemler Dönemi (2009-2015)

Orijinal raporda referans verilen Celebi ve arkadaşlarının (2009) çalışmaları ve meşhur "DullRazor" yazılımı, bu alanın temel taşlarını oluşturur.<sup>1</sup> Bu dönemdeki yaklaşımlar, ağırlıklı olarak matematiksel morfolojiye ve kural tabanlı algoritmalara dayanıyordu.

- **DullRazor ve Türevleri:** Lee ve arkadaşları tarafından 1997'de geliştirilen ve Celebi (2009) tarafından dermoskopide standartlaştırılan DullRazor, kıl tespiti için genelleştirilmiş gri tonlamalı morfolojik kapama (closing) işlemlerini kullanıyordu.<sup>5</sup> Tespit edilen kıllar daha sonra bilineer enterpolasyon ile "tıraşlanıyordu".
- **Kısıtlılıklar:** Bu yöntemler, açık ten üzerindeki kalın, koyu renkli kıllarda etkili olsa da, ince kıllar, açık renkli kıllar, damarsal yapılar veya karmaşık lezyon dokuları (örneğin pigment ağları) ile karşılaştığında başarısız oluyordu.<sup>7</sup> Daha da önemlisi, bu algoritmalar "kıl" dışındaki artefaktları (örneğin hava kabarcıkları veya cetvel izleri) ayırt edemiyor, bunları lezyonun bir parçası veya genel gürültü olarak işliyordu.<sup>8</sup> Hesaplama maliyetlerinin yüksekliği ve parametrelerin her görüntü için manuel ayarlanma gerekliliği, büyük ölçekli veri setlerinde (Big Data) kullanımlarını imkansız kılıyordu.

### 2.2 Derin Öğrenme Devrimi ve İlk CNN Modelleri (2016-2019)

2016 yılı civarında, GPU donanımlarının gelişmesi ve ImageNet gibi büyük veri setlerinin kullanıma girmesiyle birlikte dermatolojide derin öğrenme çağı başladı.

- **Esteva ve Arkadaşları (2017):** Nature dergisinde yayımlanan bu çığır açıcı çalışma, derin konvolüsyonel sinir ağlarının (CNN), klinik görüntülerin sınıflandırılmasında dermatologlarla eşdeğer veya onlardan üstün performans gösterebileceğini kanıtladı.<sup>9</sup> Bu çalışma, odağı el yapımı özniteliklerden (feature engineering) uçtan uca öğrenmeye (end-to-end learning) kaydırdı.
- **ISIC 2018 Yarışması ve Gürültü İhmali:** Codella ve arkadaşlarının (2018) ISIC 2018 yarışma özetinde belirttiği gibi, bu dönemdeki çalışmaların büyük çoğunluğu lezyon segmentasyonu ve sınıflandırmasına odaklanırken, görüntü kalitesi ve gürültü analizi büyük ölçüde göz ardı edildi.<sup>11</sup> Modeller genellikle "temiz" veriler üzerinde eğitiliyor veya çok agresif veri artırma (augmentation) yöntemleriyle gürültüye karşı dayanıklı hale getirilmeye çalışılıyordu. Ancak, gürültünün kendisini _sınıflandıran_ ve _analiz eden_ sistemler eksikti. Orijinal rapor, bu boşluğu doldurmayı amaçlayan, MobileNetV2 gibi verimli CNN'leri kullanarak gürültüyü kategorize etmeye çalışan bir geçiş dönemi çalışmasıdır.<sup>1</sup>

### 2.3 Modern Dönem: Transformerlar, Açıklanabilirlik ve Hibrit Mimariler (2020-2025)

2020'den günümüze literatürde yaşanan gelişmeler, orijinal raporun metodolojisini kökten güncelleme gerekliliğini ortaya koymaktadır.

- **Gürültü Tespiti için Transformerlar:** 2025 tarihli "Multi-label transformer for dermoscopic artifact detection" çalışması, CNN'lerin yerel özelliklere odaklanma sınırlılığını aşmak için **Classification Transformer (C-Tran)** mimarisini önermektedir.<sup>2</sup> Transformerlar, "Self-Attention" (Öz-Dikkat) mekanizmaları sayesinde görüntüdeki pikseller arası uzun menzilli ilişkileri modelleyebilir. Bu sayede, bir lezyonun üzerindeki "karanlık köşe"nin (vinyet), lezyonun kendi pigmentasyonundan farklı bir yapısal özellik olduğunu, global bağlamı kullanarak anlayabilirler.
- **EfficientNet Ailesinin Yükselişi:** MobileNetV2, mobil cihazlar için verimlilik sunmaya devam etse de, **EfficientNet (B0-B7 ve V2)** ailesi, doğruluk ve hesaplama maliyeti arasındaki dengede yeni altın standart haline gelmiştir.<sup>13</sup> Bileşik ölçekleme (compound scaling) yöntemi sayesinde EfficientNet, orijinal raporda kullanılan yüksek çözünürlüklü görüntülerden (\$1504 \\times 1129\$ px) çok daha zengin özellikler çıkarabilmektedir.<sup>13</sup>
- **Yanlılık (Bias) Kaynağı Olarak Artefaktlar:** 2023-2025 literatürünün en önemli teması, artefaktların algoritmik yanlılık kaynağı olarak tanımlanmasıdır. "Dark Corner Artifact" (DCA) üzerine yapılan çalışmalar, bu artefaktların varlığının modelin malignite olasılık skorlarını yapay olarak artırabildiğini veya azaltabildiğini göstermiştir.<sup>3</sup> Bu durum, gürültü sınıflandırmasının sadece bir "temizlik" işlemi değil, bir "etik ve güvenlik" zorunluluğu olduğunu kanıtlamaktadır.

## 3\. Dermoskopik Artefakt Taksonomisi: Klinik ve Teknik Analiz

Orijinal raporda sunulan sekiz sınıflı yapı, güncel literatür ışığında yeniden değerlendirilmeli ve genişletilmelidir. Artefaktların sınıflandırılması, sadece görsel benzerliklerine göre değil, klinik etkilerine ve derin öğrenme modelleri üzerindeki saptırıcı etkilerine göre yapılmalıdır.

### 3.1 Orijinal Sekiz Sınıfın Kritik Değerlendirmesi

- **Kıllı (Hairy):** En yaygın artefakt türüdür. Ancak literatür, "ince/açık renkli" ve "kalın/koyu renkli" kıl ayrımının önemine dikkat çekmektedir. Kalın kıllar lezyonun yapısal özelliklerini (örneğin, pigment ağını) tamamen maskelerken, ince kıllar daha çok doku analizi algoritmalarını yanıltır. Sentetik veri üretimi çalışmalarında bu iki alt tür için farklı inpainting (görüntü tamamlama) stratejileri geliştirilmiştir.<sup>15</sup>
- **Bulanık (Blurry):** Orijinal rapor bunu tek bir sınıf olarak ele alsa da, "hareket bulanıklığı" (motion blur) ile "odak kaybı" (defocus blur) teknik olarak farklıdır. Hareket bulanıklığı yönlüdür ve genellikle el titremesinden kaynaklanır; odak kaybı ise izotropiktir ve cihazın cilde tam temas etmemesinden oluşur.<sup>17</sup>
- **Hava Kabarcığı (Bubble):** İmmersiyon sıvısı (jel veya yağ) kullanıldığında oluşan hava kabarcıkları, dermoskopide "milia-benzeri kistler" (milia-like cysts) ile karıştırılabilir.<sup>19</sup> Milia-benzeri kistler, seboreik keratoz tanısında kullanılan önemli bir dermoskopik ipucudur. Bir yapay zeka modelinin hava kabarcığını kist sanması, malign bir lezyonu benign (seboreik keratoz) olarak yanlış sınıflandırmasına yol açabilir. Bu nedenle bu sınıfın doğruluğu hayati önem taşır.
- **Lisans/Metin (Licenses/Text):** Dijital filigranlar veya hasta ID'leri. Bu sınıfın tespiti ve temizlenmesi, modelin hasta mahremiyetini ihlal etmesini veya metne dayalı yanlış öğrenme (overfitting) yapmasını engellemek için zorunludur.
- **Temiz (Clean):** Referans sınıfı. Ancak "temiz" tanımı, klinik olarak yorumlanabilir kalite standartlarına (örn. netlik, aydınlatma homojenliği) dayandırılmalıdır.
- **İşaretli (Marked/Ink):** Cerrahi kalem izleri. Literatür, bu artefaktın en tehlikeli "Shortcut Learning" (Kestirme Öğrenme) kaynaklarından biri olduğunu göstermektedir. Modeller, mürekkep izini gördüğünde "bu lezyon cerrah tarafından işaretlenmiş, demek ki şüpheli/kanser" mantığını kurabilmektedir.<sup>20</sup>
- **Vinyet (Vignet/Dark Corners):** Dermoskopun fiziksel çerçevesinin oluşturduğu karartı. Bu, sadece bir görüntü hatası değil, kullanılan cihazın türünü (kontakt vs non-kontakt) belli eden bir "alan imzası"dır (domain signature). Modellerin bu imzaya dayanarak cihaz tipine göre (ve dolayısıyla o cihazın kullanıldığı hastane popülasyonuna göre) yanlı kararlar vermesi engellenmelidir.<sup>3</sup>
- **Kalem (Pen):** Genellikle lezyonu çevreleyen ince çizimler. İşaretli (Marked) sınıfından farkı, daha ince ve yapısal olmasıdır, bazen damarsal yapılarla karıştırılabilir.

### 3.2 Taksonomiye Eklenmesi Gereken Yeni Sınıflar

Raporun eksiksiz olması için, son literatürde tanımlanan şu artefakt türleri de taksonomiye dahil edilmelidir:

- **Jel Sınırları (Gel Borders):** Hava kabarcığından farklı olarak, sürülen jelin bittiği yerde oluşan ışık kırılma çizgileridir. Bu çizgiler, melanomun "pigment ağı" veya "çizgilenme" (streaks) özelliklerini taklit edebilir.<sup>2</sup>
- **Spekülar Yansıma (Specular Reflection):** Işık kaynağının (LED'lerin) cilt yüzeyinden doğrudan yansımasıyla oluşan parlak, beyaz noktalar. Bu noktalar, lezyonun renk ve doku bilgisini tamamen yok eder ve segmentasyon algoritmalarında "delik" olarak algılanabilir.<sup>23</sup>
- **Cetvel/Ölçek Çubukları (Ruler/Scale Bars):** Lezyon boyutunu göstermek için konulan cetveller. Tıpkı cerrahi mürekkep gibi, cetveller de malignite ile yüksek korelasyon gösteren bir "tehlike işareti" olarak algılanabilir.<sup>24</sup>
- **Renk Yamaları (Color Patches/Charts):** Renk kalibrasyonu için görüntüye dahil edilen renk kartları. Bunlar, lezyon dışı nesneler olarak sınıflandırılmalı ve maskelenmelidir.<sup>4</sup>

### 3.3 Çok Etiketli (Multi-Label) Sınıflandırma Gerekliliği

Orijinal rapordaki en büyük metodolojik eksiklik, "Tek Etiketli" (Single-Label) yaklaşımdır. Gerçek klinik senaryolarda bir görüntü şu özelliklere sahip olabilir: "Koyu Köşe Artefaktı (Vignet) VAR" VE "Hava Kabarcığı VAR" VE "Kıl VAR".

Orijinal model, bu görüntüyü sadece "Kıllı" olarak etiketlerse, diğer gürültü kaynaklarını (örneğin hava kabarcıklarını) görmezden gelir ve bu kabarcıklar bir sonraki aşamada lezyon analizi yapan tanı modelini yanıltabilir. Güncellenen rapor, Çok Etiketli Sınıflandırma (Multi-Label Classification) yaklaşımını benimsemeli ve çıktı katmanını buna göre (Softmax yerine Sigmoid aktivasyonu ile) yeniden tasarlamalıdır.2

## 4\. Mimari Evrim: MobileNet'ten EfficientNet ve Transformerlara Geçiş

Orijinal raporda tercih edilen **MobileNetV2** mimarisi, döneminin şartlarına göre verimli bir seçimdi. Ancak 2025 yılı perspektifinden bakıldığında, daha yüksek performans ve güvenilirlik sunan alternatifler mevcuttur. Bu bölümde, mimari seçimin neden değiştirilmesi gerektiği teknik verilerle sunulmaktadır.

### 4.1 MobileNetV2'nin Sınırlılıkları

MobileNetV2, "Inverted Residual Block" (Ters Çevrilmiş Artık Blok) ve "Linear Bottleneck" yapılarıyla mobil cihazlarda derin öğrenmeyi mümkün kılmıştır.<sup>26</sup> Yaklaşık 3.4 milyon parametre ile oldukça hafiftir ve ortalama 23ms çıkarım (inference) süresine sahiptir.<sup>28</sup> Ancak, dermoskopik gürültü sınıflandırması bağlamında şu zayıflıklara sahiptir:

- **Dikkat Mekanizması Eksikliği:** MobileNetV2, görüntünün her bölgesine eşit ağırlık verir. Oysa dermoskopik görüntülerde lezyonun kendisi ile çevresindeki gürültü (örneğin köşedeki vinyet) arasında bir önem hiyerarşisi vardır. MobileNetV2, bu ayrımı yapacak doğal bir "dikkat" (attention) mekanizmasına sahip değildir.<sup>29</sup>
- **Sınırlı Reseptif Alan:** Tamamen konvolüsyonel (CNN) yapısı nedeniyle, pikseller arasındaki yerel ilişkilere odaklanır. Görüntünün tamamına yayılmış dağınık artefaktları (örneğin tüm görüntüye yayılmış hafif bulanıklık veya dağınık jöle izleri) modellemekte, global bağlamı görebilen Transformerlara göre daha zayıftır.<sup>30</sup>

### 4.2 EfficientNetV2: Yeni Altın Standart

Raporun güncellenmiş versiyonu için önerilen birincil mimari **EfficientNetV2**'dir.

- **Fused-MBConv Blokları:** MobileNetV2'nin derinlemesine ayrılabilir konvolüsyonlarının (depthwise separable convolutions) aksine, EfficientNetV2'nin erken katmanlarında kullanılan "Fused-MBConv" yapıları, modern GPU ve TPU donanımları için optimize edilmiştir. Bu, hem eğitim süresini kısaltır hem de çıkarım hızını artırır.<sup>13</sup>
- **Bileşik Ölçekleme (Compound Scaling):** EfficientNet, ağın derinliğini, genişliğini ve çözünürlüğünü dengeli bir şekilde artırır. Orijinal çalışmada kullanılan \$1504 \\times 1129\$ piksellik yüksek çözünürlüklü görüntülerden detaylı doku özelliklerini (texture features) çıkarmak için bu ölçekleme hayati önem taşır.<sup>31</sup>
- **Performans Karşılaştırması:** ISIC veri setleri üzerinde yapılan karşılaştırmalı çalışmalarda, EfficientNet varyantları (B0-B7), MobileNetV2 ve ResNet50'ye kıyasla doğruluk, hassasiyet ve AUC (Eğri Altında Kalan Alan) metriklerinde %2 ile %5 arasında daha yüksek performans göstermektedir.<sup>14</sup> Özellikle kör görüntü kalitesi değerlendirmesinde (blind image quality assessment) EfficientNet-B0, en son teknoloji (SOTA) sonuçlara ulaşmıştır.<sup>33</sup>

### 4.3 Vizyon Transformerlar (ViT) ve Hibrit Modeller

Raporun "derin içgörü" talebini karşılamak için, **Transformer** tabanlı modellerin potansiyeli de tartışılmalıdır.

- **Global Bağımlılıklar:** Bulanıklık veya vinyet gibi artefaktlar, görüntünün geneline yayılan özelliklerdir. Transformerlar, "Self-Attention" mekanizmaları sayesinde görüntüdeki uzak noktalar arasındaki ilişkileri modelleyebilir.<sup>30</sup>
- **C-Tran ile Çok Etiketli Başarı:** Dermoskopik artefakt tespiti için özel olarak uyarlanan **C-Tran (Classification Transformer)**, etiketler arasındaki korelasyonu (örneğin, "Jel" varsa "Kabarcık" olma ihtimali yüksektir) öğrenerek, standart CNN kayıp fonksiyonlarına göre F1 skorunda 0.87 gibi çok daha yüksek bir başarıya ulaşmıştır.<sup>2</sup>
- **Öneri:** Tamamen Transformer tabanlı bir model hesaplama yükünü artırabilir (özellikle mobil dağıtım için). Bu nedenle, EfficientNetV2'nin özellik çıkarıcı (backbone) olarak kullanıldığı ve üzerine hafif bir Transformer veya Dikkat (Attention) modülünün eklendiği **Hibrit Mimariler** (örneğin MedLiteNet <sup>34</sup>) en optimal çözüm olarak önerilmektedir.

## 5\. Stratejik Metodolojik Güncellemeler

Araştırmayı uzman seviyesine taşımak için metodolojinin basit bir denetimli sınıflandırmanın ötesine geçmesi gerekmektedir.

### 5.1 Çok Etiketli Öğrenmeye Geçiş ve Kayıp Fonksiyonları

Hedef değişken \$y\$, artık \$\\{0,..., 7\\}\$ gibi tek bir sınıf indeksi değil, \$y \\in \\{0, 1\\}^8\$ şeklinde bir ikili vektör olarak tanımlanmalıdır.

- **Aktivasyon ve Kayıp:** Çıktı katmanında Softmax yerine, her sınıf için bağımsız olasılık üreten **Sigmoid** aktivasyonu kullanılmalıdır. Kayıp fonksiyonu olarak Kategorik Cross-Entropy yerine **Binary Cross-Entropy (BCE)** benimsenmelidir.
- **Dengesizlik Yönetimi:** "Temiz" sınıfı baskınken, "Kalem" veya "Metin" sınıfları nadir olabilir. Standart BCE, modeli baskın sınıfa yönlendirecektir. Bu nedenle, rapor **Focal Loss** veya daha da gelişmişi olan **Class-Adaptive Focal Loss (CAFC)** kullanımını önermelidir. CAFC, modelin çoğunluk sınıflarındaki aşırı güvenini cezalandırır ve nadir/zor örneklere odaklanmasını sağlar.<sup>2</sup>

### 5.2 Gelişmiş Veri Artırma (Augmentation) ve Sentetik Veri

Orijinal rapor standart veri artırma yöntemlerine (döndürme, çevirme) değinmiştir. Güncelleme şunları içermelidir:

- **Difüzyon Modelleri ile Gürültü Enjeksiyonu:** Sadece artefaktlı görüntüleri _bulmak_ yerine, onları _üretmek_. Son çalışmalar, Stable Diffusion gibi modellerin (LoRA ile ince ayar yapılarak) temiz lezyon görüntüleri üzerine gerçekçi kıllar, cetveller veya mürekkep izleri ekleyebildiğini göstermektedir.<sup>35</sup> Bu yöntem, mükemmel dengeli bir eğitim seti oluşturur ve aynı lezyonun "temiz" ve "kirli" hallerini kıyaslayarak (counterfactual testing) modelin gürültüye tepkisini ölçmeyi sağlar.
- **Sentetik Kıl Simülasyonu:** Kıl sınıfını eğitmek için binlerce kıl örneği toplamak yerine, farklı kalınlık, renk ve yönelimlerde sentetik kıllar üreten algoritmalar kullanılmalıdır.<sup>15</sup>

### 5.3 Kalite Değerlendirme (QA) Hattı Entegrasyonu

Nihai hedef sadece "sınıflandırma" değil, "kalite değerlendirme" olmalıdır. Sistemin çıktısı sadece "Bu görüntü Kıllı" demekle kalmamalı, bir **Kalite Skoru (Quality Score)** üretmelidir.

- **Skorlama:** Artefakt olasılıkları birleştirilerek tek bir metrik oluşturulabilir (örn. \$Q = 1 - \\max(P_{artefakt})\$).
- **Eyleme Geçirilebilir Geri Bildirim:** Sistem, klinik ortamda gerçek zamanlı çalışacak şekilde tasarlanmalıdır. Örneğin, \$P_{bulanık} > 0.8\$ ise sistem klinisyene "Görüntüyü daha iyi odaklayarak tekrar çekin" uyarısı vermelidir. \$P_{kıl} > 0.8\$ ise "Bölgeye jel uygulayın veya tıraş edin" önerisi sunulmalıdır. Bu yaklaşım, araştırmayı teorik bir sınıflandırma probleminden pratik bir klinik karar destek aracına dönüştürür.<sup>36</sup>

## 6\. Performans Metrikleri ve Değerlendirme Protokolleri

Orijinal rapordaki "%78 Doğruluk" (Accuracy) ibaresi, dengesiz veri setlerinde yanıltıcı olduğu için bilimsel bir rapor için yetersizdir.

### 6.1 Gürültü Sınıflandırması İçin Kritik Metrikler

Aşağıdaki metriklerin kullanılması şarttır:

- **F1-Skoru (Makro ve Ağırlıklı):** Özellikle "Kalem" veya "Hava Kabarcığı" gibi nadir sınıflarda Kesinlik (Precision) ve Duyarlılık (Recall) dengesini kurmak için kritik öneme sahiptir.<sup>13</sup>
- **AUC-ROC:** Her bir artefakt sınıfı için Alıcı İşletim Karakteristiği (ROC) eğrisi altındaki alanı incelemek, modelin karar eşiğinden bağımsız olarak ayırt etme gücünü gösterir.<sup>39</sup>
- **Hassasiyet-Duyarlılık (PR) Eğrileri:** Pozitif sınıfın (Artefakt) nadir olduğu durumlarda ROC'den daha bilgilendiricidir.

### 6.2 Karmaşıklık Matrisi (Confusion Matrix) Analizi

Karmaşıklık matrisi üzerinde derinlemesine bir analiz yapılmalıdır.

- **Sık Görülen Karışıklıklar:** "Bulanık" vs. "Temiz" (Hafif bulanıklığın tespiti zordur). "Kıllı" vs. "Kalem" (Her ikisi de doğrusal koyu yapılardır). "Hava Kabarcığı" vs. "Yapısız Alanlar" (Klinik özellik taklidi).
- **Sınıflar Arası Sızıntı (Leakage):** "İşaretli" (Mürekkep) bir görüntünün, aşağı yöndeki (downstream) tanı modelleri tarafından ne sıklıkla yanlışlıkla "Melanom" olarak sınıflandırıldığı nicel olarak ölçülmelidir.

### 6.3 Veri Setleri Arası Genelleme (Cross-Dataset Generalization)

Orijinal rapor sadece ISIC verilerini kullanmıştır. Modelin sağlamlığını (robustness) kanıtlamak için, **PH2** (küçük ama yüksek kaliteli), **Derm7pt** veya **PAD-UFES-20** (akıllı telefon görüntüleri) gibi harici veri setlerinde test edilmelidir. Çalışmalar, ISIC üzerinde eğitilen modellerin akıllı telefon görüntülerinde (domain shift nedeniyle) başarısız olabildiğini göstermektedir.<sup>2</sup> Bu testler, özellikle "Vinyet" ve "Bulanıklık" dedektörlerinin gerçek dünya teledermatoloji senaryolarındaki başarısını doğrulayacaktır.

## 7\. Geniş Etkiler: Yanlılık, Adalet ve Klinik Güven

Uzman seviyesindeki bir rapor, gürültü probleminin teknik boyutunun ötesine geçerek etik ve klinik sonuçlarını da ele almalıdır.

### 7.1 Artefakt Kaynaklı Yanlılık (Bias)

Derin öğrenme modelleri "tembeldir"; en kolay ayırt edici özelliği ararlar. Eğer eğitim setindeki malign melanomların %80'inde (dermatologlar şüpheli lezyonları ölçtüğü için) cetvel bulunuyorsa, model "Cetvel = Kanser" kuralını öğrenir.

- **Kanıt:** Winkler ve ark. (2021) ile Bissoto ve ark. (2020), benign görüntüler üzerine yapay olarak artefakt eklendiğinde, modelin o görüntüye verdiği malignite olasılığının arttığını deneysel olarak kanıtlamışlardır.<sup>3</sup>
- **Çözüm:** Bu raporda önerilen gürültü sınıflandırma modeli, bu sorunun _panzehiridir_. Artefaktları tespit edip işaretleyerek, bu görüntülerin tanısal model eğitiminden çıkarılmasını veya "debiasing" (yanlılık giderme) stratejileriyle (örneğin artefaktın inpainting ile silinmesi veya modele o bölgeye odaklanmaması için ceza verilmesi) temizlenmesini sağlar.

### 7.2 Açıklanabilir Yapay Zeka (XAI)

Modern bir güncelleme için **Grad-CAM** veya **Attention Maps** gibi XAI tekniklerinin entegrasyonu tartışılmaz bir gerekliliktir.

- **Görsel Doğrulama:** Modelin bir görüntüyü "Kıllı" olarak sınıflandırmasının nedeninin, gerçekten kıl tellerine odaklanması olduğunu (lezyon dokusuna değil) görsel olarak göstermek, klinisyenin sisteme olan güvenini artırır.<sup>40</sup>
- **Yorumlanabilirlik:** "Bulanık" görüntüler için XAI, bulanıklığın genel mi yoksa odaklanmış mı olduğunu göstererek, kötü çekim ile özelliksiz (featureless) lezyon arasındaki farkın anlaşılmasına yardımcı olur.<sup>30</sup>

## 8\. Önerilen Güncel Yol Haritası

Toplanan ve sentezlenen tüm veriler ışığında, "Gürültü Sınıflandırması" projesinin güncellenmesi için aşağıdaki yol haritası önerilmektedir:

### Faz 1: Veri Mühendisliği ve Genişletme

- **Veri Seti:** ISIC 2019 ve 2020 veri setleri birleştirilerek hacim artırılmalıdır.
- **Etiketleme:** Çok etiketli (multi-label) etiketleme sistemine geçilmelidir. Küçük, manuel etiketlenmiş bir setten yola çıkarak, büyük ve etiketsiz arşiv için yarı-denetimli öğrenme (örneğin "Noisy Student" eğitimi) ile etiket yayılımı (label propagation) yapılmalıdır.<sup>41</sup>
- **Veri Artırma:** Difüzyon modelleri kullanılarak sentetik "Kıllı/İşaretli/Bulanık" örnekler üreten bir "Gürültü Enjeksiyon" hattı kurulmalıdır.<sup>35</sup>

### Faz 2: Mimari Revizyon

- **Omurga (Backbone):** Standart MobileNetV2, özelliklerin daha iyi yeniden kalibrasyonu (recalibration) için Squeeze-and-Excitation bloklarına sahip **EfficientNetV2-B0** veya **MobileNetV3-Large** ile değiştirilmelidir.<sup>13</sup>
- **Çıktı Katmanı (Head):** Tek bir Softmax katmanı yerine, 8 adet Sigmoid ünitesinden oluşan çok başlı (multi-head) bir çıktı katmanı tasarlanmalıdır.
- **Kayıp Fonksiyonu:** "Temiz" ve "Kalem" sınıfları arasındaki şiddetli dengesizliği yönetmek için **Class-Adaptive Focal Loss (CAFC)** uygulanmalıdır.<sup>2</sup>

### Faz 3: Kapsamlı Değerlendirme ve Doğrulama

- **Kıyaslama (Benchmarking):** Model, ResNet50 ve ViT-Tiny gibi rakiplerle karşılaştırılmalıdır.
- **Yanlılık Kontrolü:** Modelin, yapay korelasyonlarla tasarlanmış "tuzak setlerinde" (trap sets) artefaktları ne kadar başarılı tespit ettiği ölçülmelidir.<sup>42</sup>
- **Aşağı Yönlü Etki:** Sadece gürültü tespitinin doğruluğu değil, bu gürültü filtresi uygulandığında _melanom teşhis başarısındaki artış_ (Esteva ve arkadaşlarının iddiasını doğrulayacak şekilde) ölçülmelidir.

## 9\. Sonuç

Orijinal "Noise Classification in Dermoscopic Images Using Deep Learning" raporu, otomatik dermatolojideki kritik bir darboğazı, yani girdi veri kalitesini doğru bir şekilde tespit etmiştir. Ancak alanın baş döndürücü hızı, metodoloji ve kapsamda köklü bir güncellemeyi zorunlu kılmaktadır. Tek etiketli MobileNetV2 yaklaşımından, çok etiketli, Transformer destekli (EfficientNetV2 veya C-Tran gibi) bir çerçeveye geçerek ve artefakt kaynaklı yanlılık azaltma (bias mitigation) ve difüzyon tabanlı veri üretimi gibi ileri kavramları entegre ederek, bu proje temel bir sınıflandırma görevinden, güvenilir klinik yapay zekanın temel taşına dönüşebilir.

Kıl, kabarcık ve cetvel gibi artefaktların varlığı sadece bir görüntü kirliliği değildir; hasta güvenliğini tehlikeye atabilecek tehlikeli bir yanlılık kaynağıdır. Bu nedenle, güncellenen sistem sadece veriyi "temizlemekle" kalmaz; teşhis algoritmalarının, çekim hatalarına değil, biyolojik patolojiye dayanarak karar vermesini sağlayan bir "bekçi" (gatekeeper) görevi görür. Sınıflandırmadan "kalite güvencesi ve yanlılık giderme"ye doğru bu geçiş, 2025 yılının teknoloji seviyesini (State-of-the-Art) temsil etmekte ve bu araştırmanın gelecekteki yönünü belirlemektedir.

### Tablo 1: Önerilen Model Güncellemeleri ve Karşılaştırma

| **Özellik** | **Orijinal Yaklaşım (2019 civarı)** | **Önerilen Güncel Yaklaşım (2025)** | **Gerekçe** |
| --- | --- | --- | --- |
| **Mimari** | MobileNetV2 | **EfficientNetV2 / C-Tran** | Daha iyi özellik çıkarımı, global bağlam, dikkat mekanizmaları. |
| --- | --- | --- | --- |
| **Sınıflandırma Tipi** | Tek Etiketli (Single-Label) | **Çok Etiketli (Multi-Label)** | Artefaktların eş zamanlı bulunması (örn. Kıl + Kabarcık). |
| --- | --- | --- | --- |
| **Kayıp Fonksiyonu** | Categorical Cross-Entropy | **Class-Adaptive Focal Loss (CAFC)** | Sınıf dengesizliğini (Temiz vs. Nadir Artefakt) yönetmek. |
| --- | --- | --- | --- |
| **Veri Artırma** | Klasik (Döndürme, Kırpma) | **Generative (Diffusion/GAN)** | Sentetik artefakt üretimi ile "tuzak setleri" oluşturma. |
| --- | --- | --- | --- |
| **Çıktı** | Sınıf Etiketi | **Kalite Skoru & Eylem Önerisi** | Klinik karar desteği için uygulanabilir geri bildirim. |
| --- | --- | --- | --- |
| **Açıklanabilirlik** | Yok / Sınırlı | **Grad-CAM / Attention Maps** | Modelin odaklandığı bölgeyi (artefakt mı lezyon mu) doğrulama. |
| --- | --- | --- | --- |

#### Alıntılanan çalışmalar

- Noise Classification in Dermoscopic Images Using Deep Learning -2-.pdf
- (PDF) Multi-label transformer for dermoscopic artifact detection with an imbalance-aware and confidence-penalizing loss - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/397524733_Multi-label_transformer_for_dermoscopic_artifact_detection_with_an_imbalance-aware_and_confidence-penalizing_loss>
- Dark corner artefact and diagnostic performance of a market‐approved neural network for skin cancer classification - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/351485355_Dark_corner_artefact_and_diagnostic_performance_of_a_market-approved_neural_network_for_skin_cancer_classification>
- Dermoscopic Dark Corner Artifacts Removal: Friend or Foe? - arXiv, erişim tarihi Ocak 12, 2026, <https://arxiv.org/html/2306.13446>
- \[PDF\] Approximate lesion localization in dermoscopy images - Semantic Scholar, erişim tarihi Ocak 12, 2026, <https://www.semanticscholar.org/paper/Approximate-lesion-localization-in-dermoscopy-Celebi-Iyatomi/b822b58127535a1088c774fdfb6cd31b2929d854>
- An effective hair removal algorithm for dermoscopy images - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/236125582_An_effective_hair_removal_algorithm_for_dermoscopy_images>
- A Robust Hair Segmentation and Removal Approach for Clinical Images of Skin Lesions - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/profile/Adam-Huang-4/publication/257601628_A_robust_hair_segmentation_and_removal_approach_for_clinical_images_of_skin_lesions/links/545c1ab10cf2f1dbcbcb0d26/A-robust-hair-segmentation-and-removal-approach-for-clinical-images-of-skin-lesions.pdf>
- Automatic Skin Lesion Segmentation based on Saliency and Color - SciTePress, erişim tarihi Ocak 12, 2026, <https://www.scitepress.org/Papers/2020/91449/91449.pdf>
- Esteva, A., Kuprel, B., Novoa, R.A., et al. (2017) Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks. Nature, 542, 115-118. - References - Scirp.org., erişim tarihi Ocak 12, 2026, <https://www.scirp.org/reference/referencespapers?referenceid=3275920>
- Dermatologist-level classification of skin cancer with deep neural networks - IDEAS/RePEc, erişim tarihi Ocak 12, 2026, <https://ideas.repec.org/a/nat/nature/v542y2017i7639d10.1038_nature21056.html>
- \[1902.03368\] Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC) - arXiv, erişim tarihi Ocak 12, 2026, <https://arxiv.org/abs/1902.03368>
- (PDF) Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC) - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/331034159_Skin_Lesion_Analysis_Toward_Melanoma_Detection_2018_A_Challenge_Hosted_by_the_International_Skin_Imaging_Collaboration_ISIC>
- EfficientNet-based skin cancer classification and recognition - SPIE Digital Library, erişim tarihi Ocak 12, 2026, <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13800/138001Q/EfficientNet-based-skin-cancer-classification-and-recognition/10.1117/12.3076682.full>
- COMPARATIVE ANALYSIS OF EFFICIENTNET AND RESNET MODELS IN THE CLASSIFICATION OF SKIN CANCER - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/383542189_COMPARATIVE_ANALYSIS_OF_EFFICIENTNET_AND_RESNET_MODELS_IN_THE_CLASSIFICATION_OF_SKIN_CANCER>
- Advancing dermoscopy through a synthetic hair benchmark dataset and deep learning-based hair removal - SPIE Digital Library, erişim tarihi Ocak 12, 2026, <https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-29/issue-11/116003/Advancing-dermoscopy-through-a-synthetic-hair-benchmark-dataset-and-deep/10.1117/1.JBO.29.11.116003.full>
- Black Sea Journal of Engineering and Science - DergiPark, erişim tarihi Ocak 12, 2026, <https://dergipark.org.tr/tr/download/article-file/4725613>
- No-reference Blur Assessment of Dermatological Images Acquired via Mobile Devices - SciSpace, erişim tarihi Ocak 12, 2026, <https://scispace.com/pdf/no-reference-blur-assessment-of-dermatological-images-i7x2enwxn9.pdf>
- Deep Deblurring in Teledermatology: Deep Learning Models Restore the Accuracy of Blurry Images' Classification | Request PDF - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/381765828_Deep_Deblurring_in_Teledermatology_Deep_Learning_Models_Restore_the_Accuracy_of_Blurry_Images'_Classification>
- (PDF) Artifacts and landmarks: pearls and pitfalls for in vivo reflectance confocal microscopy of the skin using the tissue-coupled device - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/336071037_Artifacts_and_landmarks_pearls_and_pitfalls_for_in_vivo_reflectance_confocal_microscopy_of_the_skin_using_the_tissue-coupled_device>
- SkinSplain: An XAI Framework for Trust Calibration in Skin Lesion Analysis - CEUR-WS.org, erişim tarihi Ocak 12, 2026, <https://ceur-ws.org/Vol-4017/paper_39.pdf>
- (PDF) Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/335176729_Association_Between_Surgical_Skin_Markings_in_Dermoscopic_Images_and_Diagnostic_Performance_of_a_Deep_Learning_Convolutional_Neural_Network_for_Melanoma_Recognition>
- Debiasing Skin Lesion Datasets and Models? Not So Fast - CVF Open Access, erişim tarihi Ocak 12, 2026, <https://openaccess.thecvf.com/content_CVPRW_2020/papers/w42/Bissoto_Debiasing_Skin_Lesion_Datasets_and_Models_Not_So_Fast_CVPRW_2020_paper.pdf>
- A soft kinetic data structure for lesion border detection - PMC - PubMed Central, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC2881363/>
- SharpRazor: Automatic removal of hair and ruler marks from dermoscopy images - PMC, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC10234178/>
- Skin Lesion Segmentation in Dermoscopic Images with Combination of YOLO and GrabCut Algorithm - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/334377779_Skin_Lesion_Segmentation_in_Dermoscopic_Images_with_Combination_of_YOLO_and_GrabCut_Algorithm>
- How to choose a pre-trained model for AI image processing? - Tencent Cloud, erişim tarihi Ocak 12, 2026, <https://www.tencentcloud.com/techpedia/125391>
- Enhancing Skin Cancer Detection and Classification in Dermoscopic Images through Concatenated MobileNetV2 and Xception Models - MDPI, erişim tarihi Ocak 12, 2026, <https://www.mdpi.com/2306-5354/10/8/979>
- Deep Learning for Melanoma Detection: A Deep Learning Approach to Differentiating Malignant Melanoma from Benign Melanocytic Nevi - PMC - NIH, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11718884/>
- Enhanced Feature Selectivity in MobileNetV2 for Skin Cancer Detection through Scaled Dot-Product Attention, erişim tarihi Ocak 12, 2026, <https://journals.flvc.org/FLAIRS/article/download/138983/144033/276001>
- Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images - arXiv, erişim tarihi Ocak 12, 2026, <https://arxiv.org/html/2509.04800v1>
- EfficientNet-Based Model for Automated Classification of Retinal Diseases Using Fundus Images - EA Journals, erişim tarihi Ocak 12, 2026, <https://eajournals.org/wp-content/uploads/sites/21/2024/11/EfficientNet-Based-Model.pdf>
- Enhancing Skin Cancer Classification Using Efficient Net B0-B7 through Convolutional Neural Networks and Transfer Learning with Patient-Specific Data - NIH, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11318802/>
- \[2509.21967\] No-Reference Image Contrast Assessment with Customized EfficientNet-B0 - arXiv, erişim tarihi Ocak 12, 2026, <https://arxiv.org/abs/2509.21967>
- MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model - arXiv, erişim tarihi Ocak 12, 2026, <https://arxiv.org/html/2509.03041v1>
- A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations - GitHub, erişim tarihi Ocak 12, 2026, <https://raw.githubusercontent.com/mlresearch/v287/main/assets/jin25b/jin25b.pdf>
- EfficientNetV2 Based Ensemble Model for Quality Estimation of Diabetic Retinopathy Images from DeepDRiD - PMC - NIH, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9955381/>
- Quantifying acceptable artefact ranges for dermatologic classification algorithms - PMC, erişim tarihi Ocak 12, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9060017/>
- Melanoma Cancer Detection Using Deep Learning and Image Processing - IJIRT, erişim tarihi Ocak 12, 2026, <https://ijirt.org/publishedpaper/IJIRT189450_PAPER.pdf>
- Multi-Class Skin Lesion Classification Using Transfer Learning with EfficientNet-B3 and Convolutional Block Attention Module - GR Journals, erişim tarihi Ocak 12, 2026, <https://www.gr-journals.com/ssc/pdf/SSC_25213.pdf>
- ScNet: a lightweight CNN with depthwise and SE modules for skin lesion classification, erişim tarihi Ocak 12, 2026, <https://www.tandfonline.com/doi/full/10.1080/21681163.2025.2576198>
- Hair removal and lesion segmentation of dermoscopic images for classification of skin cancer using deep neural networks | Request PDF - ResearchGate, erişim tarihi Ocak 12, 2026, <https://www.researchgate.net/publication/398180990_Hair_removal_and_lesion_segmentation_of_dermoscopic_images_for_classification_of_skin_cancer_using_deep_neural_networks>
- Test-Time Selection for Robust Skin Lesion Analysis, erişim tarihi Ocak 12, 2026, <https://workshop.isic-archive.com/2023/paper_bissoto.pdf>