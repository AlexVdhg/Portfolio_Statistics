### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 0cacbd00-7ac1-11f1-9a3c-8d6239cbe220
# Chargement des librairies
using PlutoUI, Distributions, Random, Plots, LinearAlgebra, StatsPlots

# ╔═╡ 3bb2a5cd-3272-4eb3-8edb-3ba84714e83f
md"""
# Portfolio -- Statistiques et calcul scientifique

### Implémentation performante d'outils statistiques : algorithme EM, échantillonnage de Gibbs, bootstrap

*Notebook Julia Pluto.jl par Alexandre Vanderhaghen*

*NB : Script développé manuellement avec l'assistance de l'outil d'autocomplétion de code et de markdown intégré à l'IDE*

---
"""

# ╔═╡ 89d63c2e-5d3d-4605-a08a-dae24a4049b6
TableOfContents(title="Table des matières", depth=3)

# ╔═╡ e882a791-d9e2-4511-b7c1-754aa7887225
Random.seed!(0)

# ╔═╡ 93b72531-ff3b-4686-a4dc-234a0ef55923
md"""
## Introduction

Comment estimer les paramètres d'un modèle quand une partie de l'information est indisponible ou cachée (latente) ? Cette problématique est le fil conducteur du présent Notebook, à travers l'exemple d'un mélange de gaussiennes aux paramètres supposés inconnus. L'objectif est de développer différents outils statistiques pour révéler de l'information latente issue d'un supposé mélange de lois ainsi que le degré d'incertitude associé aux estimations statistiques réalisées. La réflexion et les implémentations sont organisées en quatre sections comme suit :

- **Partie I - Mélange de lois gaussiennes**
- **Partie II - Estimation paramétrique ponctuelle par l'algorithme EM**
- **Partie III - Estimation paramétrique bayésienne par échantillonnage de Gibbs**
- **Partie IV - Evaluation de l'incertitude par bootstrap**
"""

# ╔═╡ fb47b9a4-4bcd-4bac-9f36-df47b12781fd
md"""
## Partie I -- Mélange de lois gaussiennes

Considérons un échantillon $x_1, \dots, x_n$ supposé provenir d'un mélange de $K$ gaussiennes tel que :

$$X_i \sim \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mu_k, \sigma_k^2), \qquad \sum_{k=1}^K \pi_k = 1$$

où $\pi_k$ est la probabilité *a priori* qu'un point vienne de la $k^{ième}$ composante.

De manière équivalente, on introduit une variable latente $z_i \in \{1, \dots, K\}$ indiquant l'origine de chaque point :

$$z_i \sim \text{Categorical}(\pi_1, \dots, \pi_K), \qquad (x_i \mid z_i) = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$$

Problème : en pratique, on observe seulement $(x_i)_{i \in [1,n]}$. Ni $(z_i)_{i \in [1,n]}$, ni $\theta = (\pi, \mu, \sigma)$ ne sont connus. L'objectif dans la suite sera donc d'estimer ces variables inconnues, suivant les paradigmes inférentiel (algorithme EM) et bayésien (échantillonnage de Gibbs).

Commençons par fixer les variables du mélange de loi. Ces valeurs constituent la vérité terrain, il s'agira des paramètres à estimer par la suite.
"""

# ╔═╡ d8ad76d4-38c4-437c-9038-256a21583765
## Paramètres du mélange
# Nombre de gaussiennes K (fixé à 2 par simplicité mais aisément généralisable)
K = 2

# ╔═╡ 54f026ee-f62e-406d-b931-9e1c6b17fd06
# Vecteur probabilité d'être généré par l'une ou l'autre des composantes
π_true = [0.3, 0.7]

# ╔═╡ 90ba3d6b-fac7-4a23-9ef3-67a3284a895a
# Vecteur moyenne des gaussiennes
μ_true = [-2.0, 3.0]

# ╔═╡ 2fab604a-fc95-4a9b-b862-16064ebf23f7
# Vecteur écart-type des lois gaussiennes
σ_true = [1.0, 1.5]

# ╔═╡ 700459f0-e2d5-4f53-a0eb-c572b166033d
# Nombre d'échantillons générés à partir du mélange de lois
n = 500

# ╔═╡ cfa897d5-ff65-4438-bcec-6b2228580648
"""
    GMM_param(n, π, μ, σ)

GMM_param simule un mélange de lois gaussiennes paramétrique et génère `n` observations indépendantes et identiquement distribuées selon `K = length(π)` composantes. Retourne le vecteur des observations `x` et le vecteur des variables latentes associées `z`.
"""
function GMM_param(n::Int,π::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    z = rand(Categorical(π), n)
    x = [rand(Normal(μ[z[i]], σ[z[i]])) for i in 1:n]
    return x, z
end

# ╔═╡ 0d6044d5-8682-42aa-895d-45e6d9a6db22
x, z_true = GMM_param(n, π_true, μ_true, σ_true);

# ╔═╡ af941db5-3830-4675-9bd2-155d206ae94f
begin
	# Définition de l'abscisse
	abscisse = range(minimum(x)-1, maximum(x)+1, length=300)

	# Tracé de l'histogramme des données simulées
	histogram(x, normalize=:pdf, bins=30, label="Données observées", color=:lightyellow, linecolor=:lightgrey, alpha=0.7)

	for k in 1:K
	    plot!(abscisse, t -> π_true[k] * pdf(Normal(μ_true[k], σ_true[k]), t),
	          label="Composante théorique $k", linewidth=2, linestyle=:dash)
	end

	# Densité théorique du mélange de lois
	mixture_density(t) = sum(π_true[k] * pdf(Normal(μ_true[k], σ_true[k]), t) for k in 1:K)
	plot!(abscisse, mixture_density, label="Densité du mélange (vérité terrain)",
	      linewidth=2, color=:teal)

	# Cadre et légende de la figure
	plot!(title="Échantillon simulé (n = $n)",
	      xlabel="x", ylabel="Densité", legend=:topleft, size=(700, 400))
end

# ╔═╡ 99a5b357-09fa-40f0-ad0c-438fa6dbc19c
md"""
L'histogramme n'épouse pas parfaitement la distribution théorique réelle des données, il n'est que la représentation d'un échantillon de 500 points tirés aléatoirement suivant cette distribution. Dans la réalité, nous ne connaîtrions pas cette distribution véritable. L'histogramme est cependant suffisamment clair pour distinguer les deux composantes gaussiennes du mélange de loi. Sous cette hypothèse de mélange de deux lois gaussiennes, l'objectif va donc être de retrouver une estimation des paramètres de chacune des gaussiennes (en pointillés rouge et vert dans la figure).
"""

# ╔═╡ de31f38d-786a-4d14-9aa6-27bef1cc56f0
md"""
## Partie II -- L'algorithme EM : une estimation paramétrique ponctuelle

L'objectif de cette partie est d'estimer les paramètres du mélange de manière ponctuelle malgré l'existence d'une variable latente inconnue grâce à l'algorithme d'Expectation-Maximization (EM).
"""

# ╔═╡ c0925072-723d-4cdf-aa03-13575f1d0272
md"""
Exprimer littéralement un mélange de loi nécessite l'intervention d'une variable latente dont le rôle est d'indiquer la composante ayant généré chaque point. De par le fait qu'elle représente une grandeur inconnue, cette variable latente est inaccessible, ce qui ne permet pas d'optimiser simplement la fonction de maximum de vraissemblance (par descente de gradient par exemple). La fonction de vraisemblance L est donnée comme suit :

$$L(\theta) = p((x_i)_{i \in [1,n]} \mid \theta) = \prod_{i=1}^n \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i \mid \mu_k, \sigma_k^2) \right).$$

Donc par passage à la log-vraisemblance afin de linéariser le produit exterieur, il vient :

$$l(\theta) = \log p((x_i)_{i \in [1,n]} \mid \theta) = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i \mid \mu_k, \sigma_k^2) \right).$$

Le logarithme d'une somme empêche toute forme fermée directe. L'algorithme EM (Expectation-Maximization) permet de contourner ce problème en itérant la mise à jour et l'évaluation des paramètres recherchés jusqu'à convergence.
"""

# ╔═╡ 852c6890-afb0-4def-8815-40ba3ebd2bec
md"""
À paramètres $\theta^{(t)}$ fixés, on calcule la probabilité *a posteriori* que le point $x_i$ provienne de la composante $k$. Celle-ci s'appelle la responsabilité de $x_i$ relativement à *k* et est dénotée $\gamma_{ik}$ :

$$\gamma_{ik} = \frac{\pi_k^{(t)} \, \mathcal{N}(x_i \mid \mu_k^{(t)}, \sigma_k^{2(t)})}{\sum_{j=1}^K \pi_j^{(t)} \, \mathcal{N}(x_i \mid \mu_j^{(t)}, \sigma_j^{2(t)})}$$

Cette expression est obtenu à partir du théorème de Bayes appliqué composante par composante. Interprétativement, on peut imaginer que chaque point "vote" pour la composante qui explique le mieux sa position, proportionnellement à sa vraisemblance pour chacune.
"""

# ╔═╡ eb8a3a37-8ce3-4550-a5bf-eebf3ae1a9f6
"""
    expectation(x, π, μ, σ)

Exécute l'étape *expectation* de l'algorithme EM en calculant la matrice des responsabilités `γ` de taille (n, k), où `γ[i, k]` est la probabilité *a posteriori* que l'observation `x[i]` provienne de la composante `k`.
"""
function expectation(x::Vector{Float64}, π::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    n = length(x)
    K = length(π)
    γ = zeros(n, K)

    # Pour chaque échantillon
    for i in 1:n
        # pour chaque composante
        for k in 1:K
            # Calcul de la responsabilité
            γ[i, k] = π[k] * pdf(Normal(μ[k], σ[k]), x[i])
        end
        # Retourne la matrice de responsabilité
        γ[i, :] ./= sum(γ[i, :])
    end

    return γ
end

# ╔═╡ 32245587-fea1-40c5-ba6f-c89640b03694
md"""
Connaissant les responsabilités à l'itération t, on peut démontrer que la $t^{ième}$ mise à jour des paramètres admet une forme close :

$$n_k = \sum_{i=1}^n \gamma_{ik} \qquad 
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i=1}^n \gamma_{ik}\, x_i, \qquad
\sigma_k^2 = \frac{1}{n_k}\sum_{i=1}^n \gamma_{ik}\,(x_i - \mu_k)^2.$$

Ici, $n_k$ est la taille estimée de la composante *k*, $\pi_k$ la probabilité d'appartenir à la composante *k*, $\mu_k$ la moyenne empirique et $\sigma_k$ l'écart-type estimés pour la composante *k*.
"""

# ╔═╡ 2efbf658-b5f9-4f5d-8e84-0d43a39b46ba
"""
    maximization(x, γ)

Met à jour les paramètres (π, μ, σ) à partir des responsabilités `γ` calculées à l'étape d'*Expectation*.
"""
function maximization(x::Vector{Float64}, γ::Matrix{Float64})
    n, K = size(γ)
    n_k = vec(sum(γ, dims=1))

    # Mise à jour de (π, μ, σ)
    π_new = n_k ./ n
    μ_new = [sum(γ[:, k] .* x) / n_k[k] for k in 1:K]
    σ_new = [sqrt(sum(γ[:, k] .* (x .- μ_new[k]).^2) / n_k[k]) for k in 1:K]

    return π_new, μ_new, σ_new
end

# ╔═╡ 0562b565-0c63-4ede-9457-524d7c4de4f6
"""
    log_likelihood(x, π, μ, σ)

Fonction chargée du calcul de la log-vraisemblance du mélange. Elle est utilisée comme critère de convergence de l'algorithme EM.
"""
function log_likelihood(x::Vector{Float64}, π::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64})
    return sum(log(sum(π[k] * pdf(Normal(μ[k], σ[k]), xi) for k in 1:length(π))) for xi in x)
end

# ╔═╡ 8c7d7d60-9327-4eca-9899-caf48f59b0ff
md"""
$$\text{Répéter jusqu'à convergence : } \quad
\theta^{(t+1)} = \underbrace{\text{Maximization}}_{\text{Màj des paramètres}}\left(\underbrace{\text{Expectation}(\theta^{(t)})}_{\text{Responsabilités}}\right)$$

L'algorithme garantit que la log-vraisemblance croît (de plus en plus faiblement) à chaque itération, ce qui permet de l'utiliser comme critère d'arrêt.
"""

# ╔═╡ 04941c27-c8ec-4bf0-a534-0cd68eb9bfa5
"""
    estimateur_EM(x, K; max_iter=100, tol=1e-6, init=nothing)

Exécute l'algorithme EM sur les données d'entrée `x` pour un mélange gaussien à `K` composantes. Retourne l'estimation des paramètres finaux ainsi que la log-vraisemblance à chaque itération, pour permettre la visualisation de la convergence.
"""
function estimateur_EM(x::Vector{Float64}, K::Int; max_iter::Int=100,
                       tol::Float64=1e-6, init::Union{Nothing,Tuple}=nothing)

    # Initialisation aléatoire si non fournie
    if init === nothing
        π = fill(1/K, K)
        μ = rand(Uniform(minimum(x), maximum(x)), K) |> sort
        σ = fill(std(x), K)
    else
        π, μ, σ = init
    end

    # Création d'un fichier de logs
    logs = (π=[copy(π)], μ=[copy(μ)], σ=[copy(σ)], l=[log_likelihood(x, π, μ, σ)])

    # Itérer jusqu'à convergence dans la limite de max_iter
    for i in 1:max_iter
        # Etape Expectation
        γ = expectation(x, π, μ, σ)
        # Etape maximisation
        π, μ, σ = maximization(x, γ)

        # Stockage des résultats
        l = log_likelihood(x, π, μ, σ)
        push!(logs.π, copy(π)); push!(logs.μ, copy(μ))
        push!(logs.σ, copy(σ)); push!(logs.l, l)

        # critère de convergence
        if abs(l - logs.l[end-1]) < tol
            break
        end
    end

    return (π=π, μ=μ, σ=σ), logs
end

# ╔═╡ fc6229b6-d9a2-4dd5-b150-900f50fc7641
θ, logs_EM = estimateur_EM(x, K; init=(π_true .+ 0.05, μ_true .+ 1.0, σ_true .+ 0.3));

# ╔═╡ 094b33dc-4cac-4d6b-8695-2c4f1f860396
begin
	n_frames = min(6, length(logs_EM.π))
	frame_idxs = round.(Int, range(1, length(logs_EM.π), length=n_frames))

	plots_list = map(frame_idxs) do t
		π_t, μ_t, σ_t = logs_EM.π[t], logs_EM.μ[t], logs_EM.σ[t]

		p = histogram(x, normalize=:pdf, bins=30, label=false,
		              color=:lightgray, linecolor=:gray, alpha=0.6)
		for k in 1:K
			plot!(p, abscisse, u -> π_t[k] * pdf(Normal(μ_t[k], σ_t[k]), u),
			      label=false, linewidth=2, color=k+1)
		end
		plot!(p, title="Itération $(t-1)", titlefontsize=9)
	end

	plot(plots_list..., layout=(2,3), size=(900, 500))
end

# ╔═╡ 1a1de9ec-b594-4e45-9299-6c23bf0673c7
plot(0:length(logs_EM.l)-1, logs_EM.l, marker=:circle, markersize=3,
     linewidth=2, xlabel="Itération", ylabel="Log-vraisemblance",
     title="Convergence de l'algorithme EM", legend=false, size=(700, 350))

# ╔═╡ 598bc9b3-df0f-4773-acab-7d360b5f1656
md"""
La log-vraisemblance croît à chaque itération, mais rien ne garantit d'atteindre le maximum global, la log-vraissemblance n'étant pas nécessairement unimodale. En outre, l'algorithme EM n'offre aucune indication sur la confiance et le degré d'incertitude à accorder à cette estimation.
"""

# ╔═╡ da7d1da7-29e8-468b-afd2-9b1060657764
md"""
## Partie III -- Échantillonnage de Gibbs : une estimation parmétrique bayésienne

L'objectif de cette section est de présenter une méthode plus avancée que l'algortihme EM pour ne plus obtenir une estimation ponctuelle des paramètres inconnus mais bien une distribution estimée de ceux-ci. Cette méthode est plus informative mais repose sur le paradigme bayésien, ce qui la rend légèrement plus complexe et demandeuse computationnellement. 
"""

# ╔═╡ 5e4f75e1-3350-4c5b-9a42-e5579015c5e4
md"""
Jusqu'ici, l'algorithme EM renvoyait un unique point $\hat\theta$. A la place, l'approche bayésienne pose des lois *a priori* sur $\theta$ et cherche la loi *a posteriori* complète resultante :

$$p(\theta \mid (x_{i})_{i \in [1, n]}) \propto p((x_{i})_{i \in [1, n]} \mid \theta) \, p(\theta)$$

Cette loi *a posteriori* n'a généralement pas de forme fermée. En particulier, elle n'en a pas pour un mélange de gaussiennes. L'échantillonnage de Gibbs offre une alternative à ce problème en tirant itérativement et à tour de rôle chaque bloc de paramètres conditionnellement à tous les autres. La séquence de tirages converge en loi vers la loi *a posteriori* recherchée (ceci se démontre en établissant la limite en loi du processus markovien associé).
"""

# ╔═╡ ba6a4eab-c991-4430-98ff-8aabdc7ebf91
md"""
Pour mettre en oeuvre l'échantillonnage de Gibbs, on commence par choisr des *priors*. Dans cet exemple, on les choisira semi-conjugués, ce qui permet à chaque étape de Gibbs de tirer directement dans une loi connue, sans étape d'acceptation-rejet :

$$\pi \sim \text{Dirichlet}(\alpha_0, \dots, \alpha_0), \qquad
\mu_k \sim \mathcal{N}(m_0, \tau_0^2), \qquad
\sigma_k^2 \sim \text{Inv-Gamma}(a_0, b_0)$$

En notant $n_k = \sum_i \mathbb{1}[z_i = k]$ et $\bar x_k$ la moyenne empirique des points affectés à la classe $k$, on obtient les quatre lois conditionnelles complètes suivantes :

**Variables latentes d'apprtenance à une classe** (identique à l'EM) :
$$z_i \mid x_i, \theta \sim \text{Categorical}(\gamma_{i1}, \dots, \gamma_{iK}), \qquad \gamma_{ik} \propto \pi_k \, \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)$$

**Pondérations dans le mélange de lois** :
$$\pi \mid z \sim \text{Dirichlet}(\alpha_0 + n_1, \dots, \alpha_0 + n_K)$$

**Moyennes** (conjugaison Normale-Normale, pour $\sigma_k^2$ fixé) :
$$\mu_k \mid x, z, \sigma_k^2 \sim \mathcal{N}(\tilde m_k, \tilde\tau_k^2), \qquad
\tilde\tau_k^2 = \left(\frac{1}{\tau_0^2} + \frac{n_k}{\sigma_k^2}\right)^{-1}, \qquad \tilde m_k = \tilde\tau_k^2\left(\frac{m_0}{\tau_0^2} + \frac{n_k \bar x_k}{\sigma_k^2}\right)$$

**Variances** (conjugaison Inverse-Gamma, à $\mu_k$ fixé) :
$$\sigma_k^2 \mid x, z, \mu_k \sim \text{Inv-Gamma}\left(a_0 + \frac{n_k}{2},\; b_0 + \frac{1}{2}\sum_{i: z_i=k} (x_i - \mu_k)^2\right)$$


La différence fondamentale entre l'approche par échantillonnage de Gibbs et l'algorithme EM repose sur le mécanisme d'affectation des échantillons à une loi parente. Même si la loi de $z_i$ ci-dessus est exactement celle de la responsabilité $\gamma_{ik}$ calculée à l'étape "Expectation" de l'algorithme EM, l'échantillonnage de Gibbs tire aléatoirement une affectation (c'est un processus stochastique qui converge) tandis que l'algorithme EM évalue une espérance d'avoir été généré par chaque compsante (c'est une étape déterministe à chaque itération ne dépendant que de l'étape de l'échantillonnage et de l'initialisation).
"""

# ╔═╡ 6fe566cf-fdd7-4bcd-9c3d-bd6689490985
md"""
S'agissant d'une méthode bayésienne, il est nécessaire de fixer les hyperpapramètres définissant les lois *a priori*. Afin de simplifier la démarche et parce que l'impact d'une loi *a priori* plutot qu'une autre n'a généralement que peu d'impact sur la loi obtenue *a posteriori* à condition d'itérer suffisamment, nous ne considérerons pas ici l'éventualité que la loi *a priori* soit définie impropre.
"""

# ╔═╡ ebaaf6c2-9bc4-4ac8-8ab5-c8f974ed0066
# Hyperparamètres
α0 = 1.0

# ╔═╡ 04729c2a-9656-47e5-af28-a979ca892d46
m0, τ0² = 0.0, 25.0 # Prior sur μ_k

# ╔═╡ 36a701a2-75be-4d6e-99b2-a0e4aa5a8fca
a0, b0  = 2.0, 2.0; # Prior sur σ_k²

# ╔═╡ fb7ff140-2073-4ed5-9420-4df41f35c8f8
"""
    iter_gibbs(x, π, μ, σ, α0, m0, τ0², a0, b0)

Réalise une itération de l'échantillonneur de Gibbs (tirage de z, π, μ, σ).
Retourne (π, μ, σ, z) mis à jour.
"""
function iter_gibbs(x::Vector{Float64}, π::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, α0::Float64, m0::Float64, τ0²::Float64, a0::Float64, b0::Float64)

    n, K = length(x), length(π)

    # Tirage des affectations z 
    γ = expectation(x, π, μ, σ)  # on réutilise la fonction de la Partie II
    z = [rand(Categorical(γ[i, :])) for i in 1:n]

    n_k = [count(==(k),z) for k in 1:K]

    # Tirage des pondéeations
    π_new = rand(Dirichlet(α0 .+ n_k))

    # Tirage de μ_k et σ_k² par composante
    μ_new = zeros(K)
    σ_new = zeros(K)

    # pour chaque composante
    for k in 1:K
        idx = findall(==(k), z)

        # Si aucun changement d'affectation
        if n_k[k] == 0
            # Alors on tire de nouvelles valeurs de μ_k et σ_k²
            μ_new[k] = rand(Normal(m0, sqrt(τ0²)))
            σ_new[k] = sqrt(rand(InverseGamma(a0, b0)))
            continue
        end

        x_k = x[idx]
        x̄_k = mean(x_k)

        # Mise à jour de μ_k à σ_k fixé
        τ̃k² = 1 / (1/τ0² + n_k[k] / σ[k]^2)
        m̃k  = τ̃k² * (m0/τ0² + n_k[k]*x̄_k / σ[k]^2)
        μ_new[k] = rand(Normal(m̃k, sqrt(τ̃k²)))

        # Mise à jour de σ_k² avec μ_k nouvellement tiré
        a_n = a0 + n_k[k]/2
        b_n = b0 + 0.5*sum((x_k .- μ_new[k]).^2)
        σ_new[k] = sqrt(rand(InverseGamma(a_n, b_n)))
    end

    return π_new, μ_new, σ_new, z
end

# ╔═╡ 193f09b2-2ef7-469a-86d1-fca1b8ae0c8a
md"""
On itère `iter_gibbs` un grand nombre de fois. Les toutes premières itérations dépendent fortement de l'initialisation (phénomène de *burn-in*) et doivent être écartées. Seules les itérations après convergence sont considérées comme des tirages approximatifs du *posterior*.
"""

# ╔═╡ 5adae2e1-c426-44ed-8dec-93231289c278
"""
    estimateur_gibbs(x, K, n_iter; burn_in, hyperparams...)

Exécute l'échantillonneur de Gibbs pendant `n_iter` itérations et retourne l'historique des tirages (π, μ, σ).
"""
function estimateur_gibbs(x::Vector{Float64}, K::Int, n_iter::Int; α0::Float64=1.0, m0::Float64=0.0, τ0²::Float64=25.0, a0::Float64=2.0, b0::Float64=2.0)

    # Initialisation
    π = fill(1/K, K)
    μ = rand(Uniform(minimum(x), maximum(x)), K) |> sort
    σ = fill(std(x), K)

    # Création d'un historique des tirages
    hist = (π=Vector{Vector{Float64}}(undef, n_iter),
            μ=Vector{Vector{Float64}}(undef, n_iter),
            σ=Vector{Vector{Float64}}(undef, n_iter))

    # Pour chaque itération t, màj de (π, μ, σ)_t
    for t in 1:n_iter
        π, μ, σ, _ = iter_gibbs(x, π, μ, σ, α0, m0, τ0², a0, b0)
        hist.π[t] = π
        hist.μ[t] = μ
        hist.σ[t] = σ
    end

    return hist
end

# ╔═╡ 73c5f99a-26ab-4c0f-87f7-c14635376f1c
begin
	Random.seed!(7)
	n_iter_gibbs = 500
	burn_in = 25

	history_gibbs = estimateur_gibbs(x, K, n_iter_gibbs; α0=α0, m0=m0, τ0²=τ0², a0=a0, b0=b0)
end;

# ╔═╡ 9b60805b-c8f7-4099-a2c7-675bfe5c11dc
begin
	μ1_chain = [history_gibbs.μ[t][1] for t in 1:n_iter_gibbs]
	μ2_chain = [history_gibbs.μ[t][2] for t in 1:n_iter_gibbs]

	plot(1:n_iter_gibbs, μ1_chain, label="μ₁", linewidth=1)
	plot!(1:n_iter_gibbs, μ2_chain, label="μ₂", linewidth=1)
	vline!([burn_in], label="Fin du burn-in", linestyle=:dash, color=:black)
	plot!(title="Convergence des estimateurs de μ₁ et μ₂", xlabel="Itération", ylabel="Valeur", size=(700, 350), legend=:outerright)
end

# ╔═╡ d8665fb0-4c73-4566-8f6b-802b12fcc7f0
begin
	μ1_post = μ1_chain[burn_in+1:end]
	θ_hat = θ
	histogram(μ1_post, normalize=:pdf, bins=40, label="Loi a posteriori de μ₁ (Gibbs)", color=:olivedrab, alpha=0.5)
	vline!([θ_hat.μ[1]], label="Estimation ponctuelle EM", linewidth=2, color=:firebrick)
	vline!([μ_true[1]], label="Vraie valeur", linewidth=3, color=:black, linestyle=:dash)
	plot!(title="Distribution à posteriori vs estimation EM", xlabel="μ₁", size=(700, 400))
end

# ╔═╡ 84f41bdb-0ffd-4c91-9353-0badf4715b23
md"""
Par rapport à l'algorithme EM qui retourne une estimation ponctuel des inconnues du mélange de lois, l'échantillonnage de Gibbs retourne une distribution complète de valeurs plausibles, au prix d'un coût computationnel légèrement supérieur et d'une convergence éventuellement plus lente si la loi *a priori* est mal choisie. Les forces de l'échantillonnage de Gibbs reposent sur sa formulation bayésienne, le rendant compatible avec tout autre algorithme bayésien, mais surtout sa capacité à transmettre davantage d'informations (en l'occurence toute une distribution), rendant par exemple possible l'évaluation des intervalles de confiance, ce que ne permettait pas l'algorithme EM.
"""

# ╔═╡ 259d5fe2-d16a-45ea-b808-5cd6647c2a9e
md"""
## Partie IV -- Bootstrap : quantification fréquentielle de l'incertitude

L'objectif de cette section est de mesurer l'incertitude sur un estimateur sans recourir à une approche bayésienne, incluant l'échantillonnage de Gibbs qui peut s'avérer ardu à mettre en oeuvre efficacement, par exemple dans le contexte du traitement de données financières en temps réél ou encore dans le cadre de l'implémentation d'algorithmes d'apprentissage automatique nécessitant l'évaluation d'un grand nombre d'estimateurs (clustering, segmentation, etc.).
"""

# ╔═╡ 3c68dbe3-7f6a-4c7e-82ae-4cde33b3bb3e
md"""
La Partie III a montré comment obtenir la distribution complète de l'estimateur $\theta$ dans un cadre bayésien coûteux computationnellement. Le Bootstrap offre une alternative plus simple et compatible avec des approches ponctuelles comme EM : l'incertitude est quantifiée par rééchantillonnage des données, sans avoir à spécifier de loi *a priori*. L'idée derrière la Bootstrap est que l'échantillon $X = (x_{i})_{i \in [1, n]}$ est une approximation discrète de la véritable loi génératrice inconnue. En rééchantillonnant avec remise dans $X$, on reproduit la variabilité qu'on observerait si l'on répétait le tirage d'échantillons dans la loi véritable un grand nombre de fois. 
"""

# ╔═╡ ce2d8aa2-3291-4619-8181-6b90028e74b7
md"""
Le principe de l'évaluation de l'incertitude par Bootstrap est relativement simple. Il s'agit de constituer $B$ sous-échantillons de taille $p$ avec remise à partir de $X$. On notera $(x^{*(b)}_{1:p})_{b \in [1, B]}$ ces sous-échantillons et $\hat\theta^{*(b)} = \text{EM}(x^{*(b)}_{1:p})$. Chaque sous-échantillon possède une variabilité artificielle simulée par l'absence ou la répétition de certains points de l'échantillon original. 

La distribution empirique des $B$ estimations $\hat\theta^{*(1)}, \dots, \hat\theta^{*(B)}$ approxime la distribution d'échantillonnage du véritable estimateur $\hat\theta$. Il est alors possible d'en déduire un intervalle de confiance :

$$\text{IC}_{95\%} = \left[\, \hat\theta^{*}_{(0.025)},\; \hat\theta^{*}_{(0.975)} \,\right].$$
"""

# ╔═╡ e34fa617-c125-4759-822a-9d1ae2956519
"""
    bootstrap_resample(x)

Tire un sous-échantillon bootstrap de `x` : un échantillon de même taille et avec remise.
"""
function bootstrap_resample(x::Vector{Float64})
    n = length(x)
    idx = rand(1:n, n)
    return x[idx]
end

# ╔═╡ 81744b9e-7950-4d4b-b8db-2331c9919925
md"""
On répète l'opération *bootstrap_resample* puis *estimateur_em* un grand nombre de fois $B$. Chaque appel à EM est indépendant des autres, ce qui permet la parallélisation/multi-threading dans le cadre d'une implémentation optimisée (ce qui n'est pas l'objet principal de ce calepin).
"""

# ╔═╡ 4bbff7b4-8a09-4ecb-bf7c-46c8b4551702
"""
    bootstrap(x, K, B; em_kwargs...)

Génère `B` sous-échantillons bootstraps de `x`, ré-estime le mélange par EM sur chacun et retourne un vecteur de `B` NamedTuples (π, μ, σ) (un par réplicat).
"""
function bootstrap(x::Vector{Float64}, K::Int, B::Int)
    results = Vector{NamedTuple}(undef, B)

    # Pour chaque sous-échantillon, évalue l'estimateur ponctuel associé par l'algorithme EM
    for b in 1:B
        x_star = bootstrap_resample(x)
        θ_star, _ = estimateur_EM(x_star, K)
        results[b] = θ_star
    end

    return results
end

# ╔═╡ 3f62a40d-52dd-4ea3-92eb-ebbe8a3832c9
begin
	Random.seed!(99)
	B = 500
	bootstrap_results = bootstrap(x, K, B)
end;

# ╔═╡ 2157239c-5b25-4300-9803-95f10da05060
md"""
Comme évoqué précedemment, rien ne force la première composante à toujours correspondre à la même gaussienne d'un sous-échantillon à l'autre. L'algorithme EM peut très bien converger vers une solution où les indices sont inversés par rapport à autre échantillon. Avant d'agréger les résultats, il faut donc imposer une convention commune (par exemple, en triant les composantes par $\mu$ croissant).
"""

# ╔═╡ 24d50ed1-b461-4047-8614-459c7037adc9
"""
    ordonnanceur(θ)

Réordonne les composantes d'un résultat EM par μ croissant pour une comparaison cohérente entre sous-échantillons bootstraps.
"""
function ordonnanceur(θ::NamedTuple)
    order = sortperm(θ.μ)
    return (π=θ.π[order], μ=θ.μ[order], σ=θ.σ[order])
end

# ╔═╡ 4b06cb3d-22ef-434d-a13b-cd5e3103b3c5
bootstrap_sorted = ordonnanceur.(bootstrap_results);

# ╔═╡ f52efd08-fe80-40f4-b086-1fb73c8fda94
begin
	μ1_bootstrap = [θ.μ[1] for θ in bootstrap_sorted]

	histogram(μ1_bootstrap, normalize=:pdf, bins=75, label="Distribution bootstrap de μ₁",
	          color=:darkorange, alpha=0.6)
	vline!([θ_hat.μ[1]], label="Estimation EM originale", linewidth=3, color=:red)
	vline!([μ_true[1]], label="Vérité terrain", linewidth=3, color=:black, linestyle=:dash)
	plot!(title="Distribution bootstrap de μ₁ (B = $B sous-échantillonnages)", xlabel="μ₁", size=(700, 400))
end

# ╔═╡ 58b34f66-3a23-4ed5-835a-a70a8da04be8
begin
	histogram(μ1_bootstrap, normalize=:pdf, bins=150, label="Bootstrap",
			  color=:darkorange, alpha=0.5)
	histogram!(μ1_post, normalize=:pdf, bins=40, label="Distribution de μ₁ par échantillonnage de Gibbs", color=:steelblue, alpha=0.5)
	vline!([μ_true[1]], label="Vraie valeur", linewidth=3, color=:firebrick, linestyle=:dash)
	plot!(title="EM et Bootstrap vs échantillonnage de Gibbs", xlabel="μ₁",
		  size=(750, 400))
end

# ╔═╡ 9f634ea9-015c-469b-92e0-bfce4d6ae796
md"""
La distribution de l'estimateur à évaluer et la plage de valeurs qu'il semble pouvoir prendre se recoupent largement quelle que soit la méthode d'évaluation considérée (EM + bootstrap ou échantillonnage de Gibbs). Dans les faits, on constate que les intervalles de confiance, pour un percentile donné, se recoupent très largement quand on considère l'une ou l'autre de ces méthodes.
"""

# ╔═╡ 83699bb2-c7f0-4d26-a7ca-a5ebbac1d3b5
md"""
## Conclusion

Finalement, à la question initiale de la possibilité de l'évaluation des estimateurs d'un mélange de lois, j'ai proposé deux méthodes d'estimation des paramètres statistiques inconnus, d'une part dans le cas d'une estimation ponctuelle simple dans le cadre de la statistique inférentielle avec l'algorithme EM (Expectation-Maximisation) et d'autre part dans le cadre de la statistique bayésienne avec la méthode d'échantillonnage de Gibbs. Une synthèse des résultats obtenus est présentée dans le Tableau 1.

*Tableau 1 - Table comparative des méthodes d'évaluation des estimateurs d'un mélange de loi et de l'incertitude associée*

| Méthodes | Estimée | Coût computationnel | Incertitude |
|---|---|---|---|
| EM seul| $\hat\theta$ | Faible (~35ms) | Aucune |
| Gibbs | $p(\theta \mid x)$ | Moyen (~150ms) | A posteriori |
| EM + Bootstrap | Distribution de $\hat\theta$ | Moyen à élevé (~2s)| Fréquentiste |

Les traveaux menés plus haut ont mis en évidence les forces et faiblesses de chacune de ces méthodes. L'approche inférentielle avec l'algorithme EM apparait comme relativement simple à implémenter et particulièrement efficace d'un point de vue calculatoire. C'est une méthode convergeant rapidement et ne nécessitant pas une puissance de calcul élevée. Elle ne permet cependant pas d'obtenir directement une estimation de l'incertitude associée à l'estimateur évalué. Pour cela, il est nécessaire de recourir à la méthode de Bootstrap qui est très coûteuse en calcul et en temps (presque 50 fois plus lente dans le cadre de notre implémentation, même s'il aurait été possible de paralléliser les calculs pour en améliorer la perormance temporelle). 

Enfin, la méthode alliant probablement le meilleur des deux approches est l'échantillonnage de Gibbs. Computationnellement, cette méthode est moins demandeuse que l'approche EM doublée d'un Bootstrap. Elle demeure cependant 5 fois plus chronophage que l'algorithme EM simple. Sa force réside dans sa nature bayésienne, la rendant certes plus complexe à appréhender (et donc à implémenter) mais également porteuse de davantage d'information (entièreté de la distribution des données). 

En conclusion, le choix de la méthode la plus appropriée à implémenter repose donc essentiellement sur le contexte : ressources disponibles (temps, puissance de calculs), besoins de scalabilité, intérêt ou non pour les incertitudes statistiques, etc. Une bonne compréhension de ces outils statistiques est un bagage nécessaire à de nombreuses disciplines, de l'optimisation de portefeuilles en finance à l'élaboration de projections climatiques en passant par la recherche algorithmique en apprentissage automatique. Ainsi, j'espère avoir su faire la démonstration de ma maîtrise de concepts fondamentaux de mathématiques et statistiques théoriques à travers ce projet, dans la perspectives où de tels compétences seraient appréciées dans le cadre de mes missions.
"""

# ╔═╡ 38620a0a-d2f1-43e7-b9f5-e6c828027728
html"""
<style>
h1 {
    border-bottom: 3px solid #4063D8;
    padding-bottom: 10px;
}
h2 {
    color: #389826;
    margin-top: 40px;
}
.chapter-sep {
    border: none;
    height: 2px;
    background: linear-gradient(to right, #9558B2, #4063D8, #389826, #CB3C33);
    margin: 50px 0;
}
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.129"
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "86c8cea6d2c3ebc66524709594f072946e1b1181"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "7f54761502ff149a9d492e4acefe9805898e29b3"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.2+0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
git-tree-sha1 = "99ee296f88c12485402e37c2fd025f95ae097637"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.9"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "cd3c5ac74cd3923c8945c6a81518c46abd0e73a3"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.129"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.Gamma]]
git-tree-sha1 = "86f86b6168a016ed88e4ae4e64577b98c3b59e8e"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.1.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "18d7deab5fb0440dc6a7b6993c5c27b25420de10"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.29"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1dae3057da6f2b9c857afef03177bbdc7c4afe92"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3ac157462e1e800777cc97d0eafd1bdb5356a470"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "21.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e8dcbeef032ba2f9051a44ac22b4e54e3a1a0099"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.6"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7c3ff68a904d0f7404e5d2f7f5bc667934d8d616"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "ca562494d657e2b69191e440e9c28b9692d67944"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.28"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "26766d4b5f1a410c218a19b85a672c6edb693c65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.40"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "ed45bcc7cf3c8887595b973f2b1efbe91dcc50ec"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "770240df9a3b8888065046948f7a09b4e0f997d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "0f38a06c83f0007bbab3cf911262841c9a0f07e0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.13.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─3bb2a5cd-3272-4eb3-8edb-3ba84714e83f
# ╠═89d63c2e-5d3d-4605-a08a-dae24a4049b6
# ╠═0cacbd00-7ac1-11f1-9a3c-8d6239cbe220
# ╠═e882a791-d9e2-4511-b7c1-754aa7887225
# ╟─93b72531-ff3b-4686-a4dc-234a0ef55923
# ╟─fb47b9a4-4bcd-4bac-9f36-df47b12781fd
# ╠═d8ad76d4-38c4-437c-9038-256a21583765
# ╠═54f026ee-f62e-406d-b931-9e1c6b17fd06
# ╠═90ba3d6b-fac7-4a23-9ef3-67a3284a895a
# ╠═2fab604a-fc95-4a9b-b862-16064ebf23f7
# ╠═700459f0-e2d5-4f53-a0eb-c572b166033d
# ╠═cfa897d5-ff65-4438-bcec-6b2228580648
# ╠═0d6044d5-8682-42aa-895d-45e6d9a6db22
# ╟─af941db5-3830-4675-9bd2-155d206ae94f
# ╟─99a5b357-09fa-40f0-ad0c-438fa6dbc19c
# ╟─de31f38d-786a-4d14-9aa6-27bef1cc56f0
# ╟─c0925072-723d-4cdf-aa03-13575f1d0272
# ╟─852c6890-afb0-4def-8815-40ba3ebd2bec
# ╠═eb8a3a37-8ce3-4550-a5bf-eebf3ae1a9f6
# ╟─32245587-fea1-40c5-ba6f-c89640b03694
# ╠═2efbf658-b5f9-4f5d-8e84-0d43a39b46ba
# ╠═0562b565-0c63-4ede-9457-524d7c4de4f6
# ╟─8c7d7d60-9327-4eca-9899-caf48f59b0ff
# ╠═04941c27-c8ec-4bf0-a534-0cd68eb9bfa5
# ╠═fc6229b6-d9a2-4dd5-b150-900f50fc7641
# ╟─094b33dc-4cac-4d6b-8695-2c4f1f860396
# ╟─1a1de9ec-b594-4e45-9299-6c23bf0673c7
# ╟─598bc9b3-df0f-4773-acab-7d360b5f1656
# ╟─da7d1da7-29e8-468b-afd2-9b1060657764
# ╟─5e4f75e1-3350-4c5b-9a42-e5579015c5e4
# ╟─ba6a4eab-c991-4430-98ff-8aabdc7ebf91
# ╟─6fe566cf-fdd7-4bcd-9c3d-bd6689490985
# ╠═ebaaf6c2-9bc4-4ac8-8ab5-c8f974ed0066
# ╠═04729c2a-9656-47e5-af28-a979ca892d46
# ╠═36a701a2-75be-4d6e-99b2-a0e4aa5a8fca
# ╠═fb7ff140-2073-4ed5-9420-4df41f35c8f8
# ╟─193f09b2-2ef7-469a-86d1-fca1b8ae0c8a
# ╠═5adae2e1-c426-44ed-8dec-93231289c278
# ╠═73c5f99a-26ab-4c0f-87f7-c14635376f1c
# ╟─9b60805b-c8f7-4099-a2c7-675bfe5c11dc
# ╟─d8665fb0-4c73-4566-8f6b-802b12fcc7f0
# ╟─84f41bdb-0ffd-4c91-9353-0badf4715b23
# ╟─259d5fe2-d16a-45ea-b808-5cd6647c2a9e
# ╟─3c68dbe3-7f6a-4c7e-82ae-4cde33b3bb3e
# ╟─ce2d8aa2-3291-4619-8181-6b90028e74b7
# ╠═e34fa617-c125-4759-822a-9d1ae2956519
# ╟─81744b9e-7950-4d4b-b8db-2331c9919925
# ╠═4bbff7b4-8a09-4ecb-bf7c-46c8b4551702
# ╠═3f62a40d-52dd-4ea3-92eb-ebbe8a3832c9
# ╟─2157239c-5b25-4300-9803-95f10da05060
# ╠═24d50ed1-b461-4047-8614-459c7037adc9
# ╠═4b06cb3d-22ef-434d-a13b-cd5e3103b3c5
# ╟─f52efd08-fe80-40f4-b086-1fb73c8fda94
# ╟─58b34f66-3a23-4ed5-835a-a70a8da04be8
# ╟─9f634ea9-015c-469b-92e0-bfce4d6ae796
# ╟─83699bb2-c7f0-4d26-a7ca-a5ebbac1d3b5
# ╟─38620a0a-d2f1-43e7-b9f5-e6c828027728
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
