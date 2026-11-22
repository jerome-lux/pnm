
= Génération de réseaux de pores aléatoires à partir de courbes d'intrusion au mercure
L'approche par réseaux de pores nécessite d'être capable de générer des réseaux qui reproduisent certaines caractéristiques géométrique (porosité, taille des étranglements) et topologique (connectivité des pores) des matériaux. Dans de nombreux travaux, des réseaux réguliers avec une connectivité fixe sont utilisés [à compléter par l'auteur] afin de comprendre certains phénomènes physiques.  Les distributions de diamètre des pores et/ou des étranglements peuvent être spécifiées par exemple en utilisant des résultats de porosimétrie mercure ou d'analyse d'image.
// TODO(auteur) : trois questions sur ce paragraphe. (1) Quelle référence citer pour
// les réseaux réguliers à connectivité fixe, et quel phénomène physique ces travaux
// ont servi à comprendre ? (2) Quelle référence citer pour les travaux qui génèrent
// des réseaux aléatoires ? (3) Faut-il citer l'injection et l'extrusion séparément,
// ou une seule source pour la distribution en volume issue de la porosimétrie au
// mercure ?
Une autre approche consiste à générer les réseaux directement à partir d'image de la microstructure. On utilise généralement des image issues de micro-tomographie aux rayons X. Des nombreux algorithmes ont été développés pour identifier les pores et étranglements. Les simulations numériques réalisées sur ces réseaux ont montré de bons accords avec les mesures expérimentales. Cette approche a cependant quelques limites : la tomographie ne permet pas de résoudre les pores de taille inférieur à environ 1 µm. De plus, plus la résolution est importante, plus le volume de l'image est petit, ce qui peut poser des problème de représentativité. Enfin, même si la tomographie est une technique de plus en plus accessible, l'ensemble du process depuis la préparation des échantillons, l’acquisition et surtout l'exploitation des images peut se révéler assez complexe.
Je présente ici une méthode de génération originale que j'ai développée pendant la thèse de Fabien Gendron @gendronCarbonatationMateriauxCimentaires2019 et que nous avons appliquée ensuite dans le cadre du projet CO#sub("2")NCRETE. L'objectif était de comprendre pourquoi la carbonatation de certaines pâtes de ciment s'accompagnait de l'augmentation de leur  coefficient de diffusivité effective. Nous ne disposions alors que de mesures de porosimétrie au mercure pour caractériser la microstructure.
// TODO(auteur) : quelles pâtes de ciment ont été concernées, quelle a été l'ampleur de
// l'augmentation du coefficient de diffusivité effective observée, et par quel moyen a-t-
// elle été mesurée ? Indiquer aussi le rôle exact de la porosimétrie au mercure dans le
// diagnostic (taille des pores mise en cause).

==== De la distribution de taille des capillaires à la distribution de taille des pores
La distribution de taille de pores obtenues par porosimétrie au mercure indique le volume de mercure injecté (intrusion) ou drainé (extrusion) pour un palier de pression, i.e un diamètre d'entrée donné. Cela permet de définir une distribution en volume de rayons capillaires. La première étape de notre approche consiste à déterminer une distribution en nombre d'éléments à partir de cette distribution en volume. On note $P_c (D_i )$ la probabilité pour qu'un capillaire possède un diamètre compris entre $D_i - Delta_i$ et $D_i + Delta_i$, où $2Delta_i$ est la plage de diamètres correspondant au saut de pression entre deux paliers de pression successifs (qui n'est généralement pas constant). La distribution en nombre d'éléments s'écrit en divisant la distribution en volume que l'on note $hat(P_c)(D_i)$ par le volume moyen $overline(V_i)$ d'un élément dans l'intervalle considéré. Pour déterminer le volume moyen on suppose :
- que les capillaires sont des cylindres et que la longueur moyenne est indépendante de leur diamètre,
- que le diamètre des éléments dans l'intervalle considéré est une variable aléatoire continue suivant une loi uniforme sur $[D_i - Delta_i ; D_i + Delta_i]$.
L'espérance du volume $upright(E)(V_i)$ dans l'intervalle s'écrit alors :
$ upright(E)(V_i) = 1/4 pi upright(E)(L_i) upright(E)(D^2) $
où $upright(E)(L_i)$ est l'espérance de la longueur des capillaires dans l'intervalle.

L'espérance de $D^2$ s'écrit :
$ upright(E)(D^2) = "Var"(D) + [upright(E)(D)]^2 $

Comme

$ upright("Var")(D) = (D_i + Delta_i - D_i + Delta_i)^2 / 12 = 1/3 Delta_i^2 $
et :
$ upright(E)(D) = D_i $

L'espérance du volume dans l'intervalle $[D_i - Delta_i ; D_i + Delta_i]$ est alors :
$ upright(E)(V_i) = 1/4 pi upright(E)(L_i) (1/3 Delta_i^2 + D_i^2) $ <Esp_V>
En notant $overline(D_i^2) = 1/3 Delta_i^2 + D_i^2$, l'expression précédente s'écrit :
$ upright(E)(V_i) = 1/4 pi upright(E)(L_i) overline(D_i^2) $
On remarque que pour des petits intervalles, le terme $1/3 Delta_i^2$ pourra éventuellement être négligé devant $D_i^2$.

Finalement la probabilité d'avoir un capillaire de diamètre $D_i$ s'écrit :
$ P_c (D_i) = (hat(P_c)(D_i)) / (1/4 pi upright(E)(L_i) overline(D_i^2) ) $

La distribution de capillaires n’est cependant pas utilisable directement ; il est en effet très délicat de placer d’abord les capillaires et ensuite les pores, sans recouvrement, surtout dans le cas d’un réseau irrégulier comme celui que nous souhaitons générer. Notre approche consiste donc à déterminer la distribution des tailles de pores qui permettra, lorsqu'on les connectera, de retrouver la distribution de capillaires imposée.

On suppose tout d'abord que deux pores $i$ et $j$ sont reliés par un capillaire de diamètre $D_(i j)$ tel que
$ D_(i j) =gamma min(D_i, D_j) $
où $gamma<1$ sera considéré comme constant en première approche. En faisant l'hypothèse que les probabilités $P_p (D_i)$ et $P(D_j)$ qu'un pore ait un diamètre de la classe $D_i$ et $D_j$ sont  indépendantes, la probabilité $P_c (D_k)$ qu'un capillaire qui relie deux pores $i$ et $j$ ait un diamètre $D_k$ s'exprime comme suit :
$ P_c (D_k) = P_p (D_i > D_j = 1/gamma D_k ) + P_p (D_j > D_i = 1/gamma D_k ) + P_p (D_i = D_j = 1/gamma D_k) $

Du fait de l'indépendance des probabilités, la probabilité $P_p (D_i > D_j = 1/gamma D_k )$ pour qu'un pore ait un diamètre $D_i$ et que sont voisin ait un diamètre $D_j=1/gamma D_k < D_i$ est le produit entre $P_p (D_j)$ et la somme de toutes les probabilités $P_p (D_i)$ pour $i>j$.  On a donc :
$ P_p (D_i > D_j = 1/gamma D_k ) = P_p (D_j) sum_(i>j)^n P_p (D_i) $
où $n$ est le nombre de classes de diamètre.

Par symétrie, on a la même expression pour $P_p (D_j > D_i = 1/gamma D_k)$

Enfin, la probabilité pour que deux pores aient un diamètre de même classe est :
$ P_p (D_i = D_j = 1/gamma D_k )=P_p (D_i)^2 $

On peut alors écrire la probabilité $P_c (D_k)$ :
$ P_c (D_k=gamma D_i) = P_p (D_i)^2 + 2 P_p (D_i) sum_(j>i)^n P_p (D_j) $ <Pc_expr_Pp>

Cette expression du second degré en $P_p (D_i)$ permet de relier les distribution de diamètre des capillaires et de diamètre des pores.

L'@Pc_expr_Pp a un discriminant toujours positif :
$ Delta = 4 (sum_(j>i)^n P_p (D_j)^2) + 4P_c (gamma D_i) $
La solution positive est alors :
$ P_p (D_i) = -sum_(j>i)^n P_p (D_j) + sqrt((sum_(j>i)^n P_p (D_j))^2 + P_c (gamma D_i)) $ <Pp_sol>

En particulier pour $i=n$, on a  $sum_(j>i)^n P_p (D_j)=0$ et donc:
$ P_p (D_n) = sqrt(P_c (gamma D_n)) $

Ce qui permet de calculer les $P_p$ par recurrence, par exemple pour $D_(n-1)$ :
$ P_p (D_(n-1)) = -sqrt(P_c (gamma D_n)) + sqrt(P_c (gamma D_n) + P_c (gamma D_(n-1))) $

La distribution des diamètres de pores est donc une fonction de la racine carrée de la distribution des diamètres des capillaires. On a la relation suivante avec la distribution en volume :
$
  P_p (1/gamma D_i) = f(sqrt(P_c (D_i))) = f(sqrt((hat(P_c) (D_i))/ (1/4 pi upright(E)(L_i) overline(D_i^2))))
$ <P_p_final>

Précédemment, nous avons supposé que $upright(E)(L_i)$ était constante pour chaque classe de rayon. Afin de permettre un calage avec les données expérimentales, on supposera que $upright(E)(L_i)$ est une fonction du diamètre $D_i$ du type :
$ upright(E)(L_i)=D_i^alpha $

Une valeur de $alpha>1$ va augmenter la probabilité des petits pores. On notera que la relation de récurrence @Pp_sol assure que $sum P_p=1$.

==== Génération du réseau de pores

La première étape de la génération consiste à générer $n_p$ valeurs de diamètres de pores en utilisant la loi de répartition donnée par l'@P_p_final  et l'expression de $P_c (D_i)$:
$ P_c (D_i) = (hat(P_c)(D_i))/ (1/4 pi overline(D_i^2) D_i^alpha ) $
Si on suppose que $overline(D_i^2) tilde.eq D_i^2$ (cf. @Esp_V), on a :
$ P_c (D_i) = (hat(P_c)(D_i))/ (1/4 pi D_i^(2+alpha) ) $

Pour se faire on utilise la méthode de la transformée inverse (_inverse sampling_) où la fonction de répartition inverse est calculée numériquement par interpolation de la distribution cumulée provenant de l'@Pp_sol.

Les pores sont ensuite placés l'un après l'autre sans recouvrement par ordre croissant de diamètre dans un volume de dimensions permettant d’atteindre la porosité visée. Pour chaque pore, on tire aléatoirement la coordonnée du centre et on vérifie s'il ya recouvrement en calculant les distances avec l'ensemble des centres des pores situés à une distance inférieure à la somme du plus grand rayon et du rayon en cours. Une nouvelle coordonnée est tirée jusqu'au placement du pore (ou au bout d'un certains nombre de tentatives).

La seconde étape consiste à créer les capillaires. Comme les distributions de taille de pores sont souvent très étalées dans les milieux cimentaires (de quelques dizaines de nanomètres à des pores de l'ordre du millimètre), le réseau ne va souvent contenir qu'un nombre restreint de pores de grande taille, à moins de générer des réseaux de plusieurs centaines de milliers de pores ou plus. Pour forcer un peu le destin et faire apparaître des capillaires de grands diamètres, on peut favoriser les créations de capillaires entre les  pores de grande taille. La création des capillaires se fait en parcourant les pores par ordre décroissant de diamètre. Le processus est constitué de deux passes :
1. Lors de la première passe, on s'assure que chaque pore est connecté à au moins $c_min$ voisins où $c_min$ est la connectivité minimum des pores et on favorise les connections entre les proches voisins de plus gros diamètre. Pour se faire, on cherche l'ensemble des $n_p$ premiers voisins du pore en cours et on les classe par ordre de diamètre décroissant. On relie alors les $c_(min)$ premiers pores. Cette première passe assure que chaque pore va être relié au minimum à $c_min$ de ses plus grands proches voisins
2. Lors de la seconde passe, on connecte les voisins par ordre de distance croissante jusqu'à ce que la somme des surfaces des capillaires connectés au pore en cours atteigne la surface de recouvrement maximum $S_(max)^r$ qui est un paramètre fixé empiriquement permettant de contrôler la connectivité moyenne du réseau. Cette approche permet de ne pas fixer de connectivité a priori et de tenir compte naturellement du fait que les pores de plus grande taille peuvent avoir une connectivité très importante.  Afin d'ajouter un aspect aléatoire, la surface maximum de recouvrement est ici générée à partir d'une distribution gaussienne centrée sur un paramètre d'entrée $S_"moy"^r$

Afin de rendre le réseau périodique, les pores qui se situent proche des bords (i.e. lorsque la distance entre la surface du pore et une face est inférieur à un paramètre fixé) sont connectés aux pores de la face opposée.

L'avantage de l'algorithme proposé est qu'il ne nécessite que deux de paramètres de calage : la surface de recouvrement moyenne qui va piloter la connectivité et le paramètre $alpha$ qui va permettre d'obtenir une distribution de capillaire la plus proche possible de la distribution des diamètres d'étranglements obtenue par la porosimétrie mercure. A noter que le paramètre $gamma$ qui fixe le rapport entre le diamètre du capillaire et le diamètre du pore ne change pas la distribution du diamètre des capillaires, mais seulement celle des pores. $gamma$  pilote la proportion de la porosité contenue dans ces derniers par rapport à la porosité contenue dans les capillaires.

En pratique une valeur de $alpha=1.5$ permet de trouver un très bon accord avec la distribution des étranglements obtenus par la porosimétrie mercure. Une distribution des surfaces maximum de recouvrement avec un valeur  $S_"moy"^r=0,4$ et un écart type de  0,15 conduit À une connectivité moyenne entre 6 et 7.
