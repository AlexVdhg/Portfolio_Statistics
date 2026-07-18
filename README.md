# Portfolio_Statistics

[![Julia](https://img.shields.io/badge/Julia-1.10-9558B2?style=flat&logo=julia&logoColor=white)](https://julialang.org/)
[![Pluto.jl](https://img.shields.io/badge/Pluto.jl-Notebook-4063D8?style=flat)](https://plutojl.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Français

### Description

Ce projet est le premier d'une série qui constitue mon portfolio. Il prend la forme d'un Notebook Pluto et illustre des concepts statistiques avancés de manière accessible. L'ensemble du projet est développé en Julia, un langage de programmation open-source, gagnant rapidement en papularité et particulièrement adapté au calcul scientifique.

Ce notebook aborde l'estimation des paramètres d'un mélange de lois gaussiennes à partir de données observées, en s'appuyant notamment sur l'algorithme EM dans le cadre de l'inférence fréquentielle ou l'échantillonnage de Gibbs (Gibbs sampling)  pour le paradigme bayésien.

### Lire le notebook sans installer Julia

Aucune installation n'est nécessaire pour consulter le notebook : [Lien de lecture du notebook](*[lien](https://pluto.land/n/ru2vdu6i)*)

### Contexte et objectifs

L'objectif de ce projet est double :
- Pédagogique : vulgariser des concepts statistiques avancées (mélanges gaussiens, algorithme EM, échantillonnage de Gibbs).
- Démonstratif : illustrer ma capacité à modéliser un problème mathématique, l'implémenter en Julia et communiquer les résultats associés.

### Installation

Si vous souhaitez exécuter le notebook localement plutôt que d'utiliser le lien de lecture :

```bash
# Cloner le dépôt
git clone https://github.com/AlexVdhg/Portfolio_Statistics.git
cd Portfolio_Statistics

# Lancer Julia
julia
```

Puis, dans le REPL Julia :

```julia
using Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run()
```

Ouvrez ensuite le fichier `.jl` du notebook depuis l'interface Pluto qui s'ouvre dans votre navigateur.

Une fois le notebook ouvert (via le lien de lecture ou localement), les cellules sont exécutées automatiquement par Pluto et sont interactives : vous pouvez modifier les paramètres (nombre d'itérations, taille de l'échantillon, etc.) pour observer leur impact en temps réel.

### Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## English

### Description

This project is the first in a series that makes up my portfolio. It takes the form of a Pluto notebook that explains advanced statistical concepts in accessible terms. The entire project is developed in Julia, a modern, open-source programming language that is rapidly gaining traction and is particularly well-suited for scientific computing.

This notebook covers parameter estimation for a Gaussian mixture model from observed data, relying in particular on the Expectation-Maximization algorithm and the Gibbs sampling algorithm for Bayesian inference.

### Read the notebook without installing Julia

No installation is required to view the notebook: [Notebook viewing link](*[lien](https://pluto.land/n/ru2vdu6i)*)

### Background and Objectives

This project has two main goals:
- Educational: make advanced statistical concepts (Gaussian mixtures, Bayesian inference, Gibbs sampling) accessible.
- Demonstrative: showcase my ability to model a statistical problem, implement it in Julia, and communicate results clearly.

### Installation

If you'd like to run the notebook locally instead of using the read-only link:

```bash
# Clone the repository
git clone https://github.com/AlexVdhg/Portfolio_Statistics.git
cd Portfolio_Statistics

# Launch Julia
julia
```

Then, in the Julia REPL:

```julia
using Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run()
```

Open the notebook's `.jl` file from the Pluto interface that opens in your browser.

Once the notebook is open (via the read-only link or locally), cells are automatically run by Pluto and are interactive: you can adjust parameters (number of iterations, sample size, etc.) to observe their impact in real time.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
