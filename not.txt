Önemli !

Cbow ve Skip-gram modellerini kullanırken sınıflandırıcı olarak Multinomial Naive Bayes kullanımda X değerleri negatif olabildiği için hata alındı.
Çözüm yöntemleri :
1. Multinomial Naive Bayes sınıflandırıcısını atlamak.
2. MNB yerine Gausssian Naive Bayes kullanmak.
3. X değerlerini [0,1] değerleri arasına normalize etmek.

Bu projede çözüm olarak normalize etme yöntemi kullanılmıştır.

