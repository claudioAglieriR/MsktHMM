
.packageName <- 'EMMIXskew'


################################################################################
################################################################################
################################################################################
################################################################################

# 1)

################################################################################
################################################################################
################################################################################
################################################################################



################################################################################
# EmSkew
# ------------------------------------------------------------------------------
# Funzione wrapper di alto livello per il fitting di modelli di miscele finite
# di variabili latenti *skew* multivariate (Normal, t‑Student, Skew‑Normal,
# Skew‑t). Si interfaccia a routine di livello inferiore (in R e C) che
# eseguono l’algoritmo EM vero e proprio.
#
# L’utente può:
#   • Fornire partizioni iniziali (`clust`)
#   • Fornire parametri iniziali completi (`init`)
#   • Delegare completamente la scelta degli start (multi‑start automatico)
#
# A seconda dei casi vengono invocate:
#   • EmSkewfit2 → fitting con parametri iniziali noti
#   • EmSkewfit1 → fitting con partizione iniziale nota
#   • init.mix   → generazione automatica di più start (k‑means, random, h‑clust)
#
# Dopo il fitting calcola:
#   • Codice di errore (con messaggi esplicativi)
#   • Criterio ICL per la selezione del modello
#   • “Mode points” di ogni componente via EmSkewMOD
#   • Eventuale stampa di riepilogo se `debug = TRUE`
#
# Riferimento teorico principale: Sahu, Dey & Branco (2003) – distribuzione
# Skew‑t multivariata non ristretta; l’algoritmo EM è un’estensione per miscele.
################################################################################

EmSkew <- function(dat,                     # matrice (n x p) dei dati
                   g,                       # numero di componenti di miscela
                   distr   = "mvn",         # tipo di distribuzione di base
                   ncov    = 3,             # struttura di Σ (1: comune, 2: diag, 3: libera, …)
                   clust   = NULL,          # etichette di partizione iniziale (opz.)
                   init    = NULL,          # lista di parametri iniziali (opz.)
                   itmax   = 1000,          # max iterazioni dell’EM
                   epsilon = 1e-6,          # soglia di convergenza log‑lik
                   nkmeans = 0,             # n° partizioni k‑means per multi‑start
                   nrandom = 10,            # n° partizioni random per multi‑start
                   nhclust = FALSE,         # TRUE: usa anche gerarchico
                   debug   = TRUE,          # stampa messaggi di debug
                   initloop = 20)           # loop intern. per pacchetto inizializzazione
{
  dat <- as.matrix(dat)                     # forza a matrice per sicurezza
  browser()
  ##############################################################################
  # 1) Scelta del percorso di inizializzazione
  # ----------------------------------------------------------------------------
  # Priorità:
  #   a) parametri completi (`init`)          → EmSkewfit2
  #   b) partizione (`clust`)                 → EmSkewfit1
  #   c) nessuno dei due                      → init.mix → EmSkewfit2
  ##############################################################################
  
  if (!is.null(init) || !missing(init)) {
    
    # Caso (a): parametri già noti → fitting diretto
    obj <- EmSkewfit2(dat, g, init, distr, ncov, itmax, epsilon)
    
  } else {
    
    if (is.null(clust) || missing(clust)) {
      
      # Caso (c): nessuna info → genera multi‑start automatici
      init <- try(
        init.mix(dat, g, distr, ncov,
                 nkmeans, nrandom, nhclust, initloop)
      )
      
      if (!is.null(init)) {
        # Se l’inizializzazione è andata a buon fine
        obj <- EmSkewfit2(dat, g, init, distr, ncov, itmax, epsilon)
      } else {
        # Nessun start valido trovato → interrompe con codice errore 20
        warning("not find initial values")
        obj         <- list()
        obj$error   <- 20
      }
      
    } else {
      # Caso (b): partizione nota → stima parametri iniziali interni
      obj <- EmSkewfit1(dat, g, clust, distr, ncov,
                        itmax, epsilon, initloop)
    }
  }
  
  ##############################################################################
  # 2) Gestione del codice di errore restituito dalla routine di fitting
  # ----------------------------------------------------------------------------
  #  0 : convergenza regolare
  #  1 : non converge entro itmax
  #  2 : densità non calcolabile in E‑step iniziale
  #  3 : allocazione non valida in E‑step iniziale
  # 12 : densità non calcolabile durante l’EM
  # 13 : allocazione non valida durante l’EM
  # 20 : impossibile trovare valori iniziali
  ##############################################################################

  error <- obj$error
  ret   <- NULL
  
  msg <- switch(tolower(error),
                '1'  = paste("stopped at (did not converge within)", itmax, "iterations"),
                '2'  = "density fails at initial steps!",
                '3'  = "allocation fails at initial steps",
                '12' = "density fails at estps!",
                '13' = "allocation fails at estep",
                '20' = "not find initials"
  )
  
  if (error > 1) {
    cat('\n-----------------------\n')
    warning("error code = ", error, '\n', msg, "\n")
    cat('\n-----------------------\n\n')
  }
  
  ##############################################################################
  # 3) Riassunto finale – solo se la stima è valida (error <= 1)
  ##############################################################################
  
  if (error <= 1) {
    
    # 3a) Calcolo dell’Information Criterion ICL
    ICL <- getICL(dat, nrow(dat), ncol(dat), g, distr, ncov,
                  obj$pro,    # pesi di miscela
                  obj$mu,     # medie
                  obj$sigma,  # matrici di cov.
                  obj$dof,    # gradi di libertà (se t / skew‑t)
                  obj$delta,  # vettori di asimmetria (se skew‑*)
                  obj$clust)  # classificazione rigida
    
    # 3b) Ricerca del “mode point” di ciascuna componente
    modpts <- EmSkewMOD(ncol(dat), g, distr,
                        obj$mu, obj$sigma, obj$dof, obj$delta)
    
    # 3c) Raccoglie tutto in un unico oggetto di ritorno
    ret         <- obj
    ret$ICL     <- ICL$ICL
    ret$modpts  <- modpts
    
    # 3d) Output di debug leggibile dall’utente
    if (debug) {
      
      # Messaggio sul tipo di modello utilizzato
      msg <- switch(tolower(distr),
                    'mvn' = paste(g, "- Component Multivariate Normal Mixture Model"),
                    'mvt' = paste(g, "- Component Multivariate t      Mixture Model"),
                    'msn' = paste(g, "- Component Multivariate Skew Normal Mixture Model"),
                    'mst' = paste(g, "- Component Multivariate Skew‑t Mixture Model")
      )
      
      cat('\n-----------------------\n\n')
      cat(msg, "\n")
      cat('\n-----------------------\n\n')
      
      # Stampa selettiva dei parametri stimati
      switch(tolower(distr),
             'mvn' = print(obj[1:8]),                # pro, mu, sigma, clust, …
             'mvt' = print(obj[1:9]),                # + dof
             'msn' = print(obj[c(1:8,10)]),          # + delta
             'mst' = print(obj[1:10])                # + dof, delta
      )
      # Stampa dell’ICL
      print(ICL)
      cat('\n-----------------------\n')
    }
  }

  # 4) Valore di ritorno: lista completa in caso di successo,
  #    NULL altrimenti (o lista con errore se >1)
  ret
}


################################################################################
################################################################################
################################################################################
################################################################################

# 2)

################################################################################
################################################################################
################################################################################
################################################################################




################################################################################
# init.mix
# ------------------------------------------------------------------------------
# Routine di *multi‑start initialization* per modelli di miscele finite.
# Dato un set di dati `dat` (n x p) e il numero di componenti `g`,
# costruisce varie partizioni iniziali (k‑means, random, gerarchiche)
# e, per ognuna, stima in maniera rapida i parametri del modello tramite
# `initEmmix()`.  Restituisce l’oggetto con la log‑verosimiglianza più alta
# (campo `$loglik`).  In caso di fallimento completo ritorna `NULL`.
#
# Argomenti
# ---------
# dat      : matrice dei dati (n osservazioni × p variabili)
# g        : numero di componenti di miscela desiderato
# distr    : distribuzione di base ("mvn", "mvt", "msn", "mst")
# ncov     : indice struttura di covarianza (1 = comune, 2 = diag, 3 = libera,…)
# nkmeans  : quante partizioni k‑means generare (0 ⇒ salta k‑means)
# nrandom  : quante partizioni casuali generare
# nhclust  : se TRUE valuta anche partizioni da clustering gerarchico
# maxloop  : max iterazioni interne usate da `initEmmix()` per affinare
#            i parametri (default 20)
################################################################################

init.mix <- function(dat, g,
                     distr, ncov,
                     nkmeans, nrandom,
                     nhclust,
                     maxloop = 20)
{
  ##############################
  # 0) Inizializzazioni generali
  ##############################
  
  browser()
  found            <- list()          # conterrà il *best start* trovato
  found$loglik     <- -Inf            # log‑likelihood peggiore possibile

  n       <- nrow(dat)                # numero di osservazioni
  clust   <- rep(1, n)                # partizione fittizia (tutto nel gruppo 1)
  mclust  <- NULL                     # migliore partizione candidata
  
  ###########################################################################
  # 1) Caso g > 1 (vere miscele) → ricerca multi‑start
  ###########################################################################
  if (g > 1) {
    
    # ----------------------------------------------------------------------
    # 1a) Partizioni K‑means
    # ----------------------------------------------------------------------
    if (nkmeans > 0) {
      
      for (i in 1:nkmeans) {
        
        # Esegue k‑means con 5 nstart e salva le etichette
        clust <- kmeans(dat, g, nstart = 5)$cluster
        
        # Scarta partizioni che producono gruppi troppo piccoli (<10)
        if (min(table(clust)) < 10) next
        
        # Stima parametri rapidi basati su questa partizione
        obj <- try(initEmmix(dat, g, clust,
                             distr, ncov, maxloop))
        
        # Verifica struttura corretta dell’output e assenza di errori
        if (length(obj) != 8 || obj$error) next
        
        # Aggiorna miglior soluzione se log‑lik ↑
        if (obj$loglik > found$loglik) {
          found  <- obj
          mclust <- clust
        }
      } # fine ciclo k‑means
      
      # Se k‑means non ha fornito nulla di valido, intensifica random start
      if (is.null(mclust))
        nrandom <- 10
    }

    # ----------------------------------------------------------------------
    # 1b) Partizioni Random
    # ----------------------------------------------------------------------
    if (nrandom > 0)
      for (i in 1:nrandom) {
        
        # Assegna a caso ogni osservazione ad uno dei g gruppi
        clust <- sample(1:g, n, replace = TRUE)
        
        obj <- try(initEmmix(dat, g, clust,
                             distr, ncov, maxloop))
        
        if (length(obj) != 8 || obj$error != 0) next
        
        if (obj$loglik > found$loglik) {
          found  <- obj
          mclust <- clust
        }
      }
    
    # ----------------------------------------------------------------------
    # 1c) Clustering Gerarchico (opzionale)
    # ----------------------------------------------------------------------
    # Nota: si utilizza come “distanza” 1‑correlazione fra osservazioni
    methods <- c("complete")           # altri metodi: ward, single, ...
    
    if (nhclust) {
      
      # Matrice di distanza basata sulle correlazioni tra righe di `dat`
      dd <- as.dist((1 - cor(t(dat))) / 2)
      
      for (j in methods) {
        
        clust <- cutree(hclust(dd, j), k = g)
        
        obj <- try(initEmmix(dat, g, clust,
                             distr, ncov, maxloop))
        
        if (length(obj) != 8 || obj$error != 0) next
        
        if (obj$loglik > found$loglik) {
          found  <- obj
          mclust <- clust
        }
      } # fine ciclo metodi gerarchici
    }   # fine blocco nhclust
    
  } else {  ###################################################################
    # 2) Caso g == 1 → nessuna miscela, un’unica componente
    ###########################################################################
    
    obj <- try(initEmmix(dat, g, clust,
                         distr, ncov, maxloop))
    
    # Se l’output rispetta la struttura attesa lo accetta
    if (length(obj) == 8) {
      found  <- obj
      mclust <- clust
    }
  }
  
  ###########################################################################
  # 3) Check finale: se nessuna partizione è risultata valida
  ###########################################################################
  if (is.null(mclust)) {
    found <- NULL
    warning("failed to find initial values!")
  }
  
  ###########################################################################
  # 4) Ritorna l’oggetto migliore (o NULL in caso di fallimento)
  ###########################################################################
  found
}


################################################################################
################################################################################
################################################################################
################################################################################

# 3)

################################################################################
################################################################################
################################################################################
################################################################################



################################################################################
# initEmmix
# ------------------------------------------------------------------------------
# Stima **rapida** (pochi passi EM o metodologie closed‑form) dei parametri
# iniziali per un modello di mistura g‑componenti, a partire da una *partizione*
# fissa `clust`.  È il cuore della strategia multi‑start: valutiamo centinaia di
# partizioni ma ciascuna viene raffinata solo per un massimo di `maxloop`
# iterazioni, così da non spendere tempo eccessivo su start potenzialmente
# scadenti.
#
# Internamente chiama la funzione C `initfit_` (compilata nel package
# **EMMIXskew**) che implementa:
#   • E‑step  ridotto e chiuso (utilizzando momenti troncati delle t multivariate)
#   • M‑step  esplicito per pro, μ, Σ, δ, ν (a seconda del modello scelto)
#
# L’output di `initfit_` è una lista con vettori *flattened*; qui convertiamo in
# array R di forma (p × g) o (p × p × g), poi restituiamo la lista.
#
# Argomenti
# ---------
# dat      : matrice (n × p) dei dati
# g        : numero di componenti
# clust    : vettore di etichette (lunghezza n, valori 1…g)
# distr    : "mvn", "mvt", "msn", "mst"
# ncov     : struttura di Σ (1 = comune, 2 = diag, 3 = libera, 4 = spherical,
#                                 5 = comune‑diag)
# maxloop  : iterazioni massime concesse a `initfit_`
################################################################################

initEmmix <- function(dat, g, clust,
                      distr, ncov,
                      maxloop = 20)
{
  # ---------------------------------------------------------------------------
  # 0) Mapping distribuzione → indice intero che la routine C si aspetta
  #    (più veloce che passare stringhe)
  # ---------------------------------------------------------------------------
  ndist <- switch(tolower(distr),
                  "mvn" = 1,  # Multivariate Normal
                  "mvt" = 2,  # Multivariate t
                  "msn" = 3,  # Multivariate Skew‑Normal (Azzalini)
                  "mst" = 4,  # Multivariate Skew‑t     (Sahu‑Dey‑Branco)
                  5)          # codice inesistente → verrà intercettato sotto
  
  browser()
  
  # Verifica disponibilità del modello richiesto
  if (ndist > 4 || ncov < 1 || ncov > 5)
    stop("the model specified is not available yet")
  
  # ---------------------------------------------------------------------------
  # 1) Preparazione dei dati e variabili ausiliarie
  # ---------------------------------------------------------------------------
  dat <- as.matrix(dat)            # garantiamo struttura matrice
  n   <- nrow(dat)                 # numero osservazioni
  p   <- ncol(dat)                 # dimensione variabile multivariata
  

  # `clust` deve essere un intero 1…g; unclass+as.ordered assicura ciò
  clust <- unclass(as.ordered(clust))
  
  # ---------------------------------------------------------------------------
  # 2) Chiamata alla routine C (`initfit_`)
  #    NB: gli argomenti passati per *reference* verranno popolati in‑place.
  # ---------------------------------------------------------------------------
  obj <- .C(
    'initfit_',                    # nome simbolo in libreria condivisa
    PACKAGE = "EMMIXskew",         # pacchetto che lo contiene
    as.double(dat),                #  1: dati, vettore column‑major
    as.integer(n),                 #  2: n
    as.integer(p),                 #  3: p
    as.integer(g),                 #  4: # componenti
    as.integer(ncov),              #  5: struttura Σ
    as.integer(ndist),             #  6: codice distribuzione
    pro    = double(g),            #  7: mixing proportions (π_k)
    mu     = double(p * g),        #  8: medie flatten (p×g)
    sigma  = double(p * p * g),    #  9: covarianze flatten (p×p×g)
    dof    = double(g),            # 10: ν_k (solo per t / skew‑t)
    delta  = double(p * g),        # 11: vettori di asimmetria δ_k (skew‑*)
    # -------- variabili d’appoggio per E‑step (occupano slot 12‑23) ----------
    tau    = double(n * g),        # 12: posteriori τ_ik
    double(n * g),                 # 13: v_ik (per t)
    double(n * g),                 # 14: z_ik (indicatori skew)
    double(n * g),                 # 15: log‑densità individuali
    double(n * g),                 # 16: … (placeholder per eventuali update)
    sumtau = double(g),            # 17
    sumvt  = double(g),            # 18
    sumzt  = double(g),            # 19
    sumlnv = double(g),            # 20
    ewy    = double(p * g),        # 21
    ewz    = double(p * g),        # 22
    ewyy   = double(p * p * g),    # 23
    loglik = double(1),            # 24: log‑verosimiglianza corrente
    as.integer(clust),             # 25: partizione fissa
    error  = integer(1),           # 26: codice di errore 0 = OK
    as.integer(maxloop)            # 27: # iterazioni max
  )[c(7:11, 24, 26)]               # ← *subset* dei campi che ci servono
  
  # ---------------------------------------------------------------------------
  # 3) Post‑processing: costruzione lista R di ritorno
  # ---------------------------------------------------------------------------
  error <- obj$error     # 0 = successo; >0 = qualche problema
  ret   <- NULL

  if (error == 0) {
    # Ri‑impacchettiamo vettori flatten in array 3‑D o 2‑D
    ret <- list(
      distr = distr,
      error = error,
      loglik = obj$loglik,
      
      pro  = obj$pro,                                # (g)
      mu   = array(obj$mu,    c(p, g)),              # (p × g)
      sigma= array(obj$sigma, c(p, p, g)),           # (p × p × g)
      dof  = obj$dof,                                # (g)   (NA se non usata)
      delta= array(obj$delta, c(p, g))               # (p × g) (NA se non usata)
    )
  } else {
    warning("error: ", error)  # propagate messaggio di errore della C‑routine
  }
  
  return(ret)
}





################################################################################
################################################################################
################################################################################
################################################################################

# 4)

################################################################################
################################################################################
################################################################################
################################################################################



################################################################################
# EmSkewfit2
# ------------------------------------------------------------------------------
# Esegue l’algoritmo EM **completo** (fino a convergenza) per un modello di
# miscela skew multivariata, partendo da un set di parametri iniziali *già
# forniti* dall’utente o da `init.mix`.  È l’entry‑point “di lavoro” quando
# l’inizializzazione è nota: rispetto a `initEmmix` non fa shortcut ma porta
# l’EM fino ad uno dei criteri di arresto:
#     • log‑verosimiglianza varia < `epsilon`
#     • numero massimo di iterazioni `itmax`
#
# Implementazione “pesante” in linguaggio C (`emskewfit2`) richiamata via .C:
# quella routine si occupa di:
#   • E‑step con calcolo momenti troncati v_i, z_i, τ_ik
#   • M‑step chiuso per pro, μ_k, Σ_k, δ_k, ν_k
#   • Aggiornamento di loglik, AIC/BIC e cluster MAP
#
# Argomenti
# ---------
# dat      : matrice dati (n × p)
# g        : # componenti di miscela
# init     : lista parametri iniziali (pro, mu, sigma, dof, delta)
# distr    : "mvn", "mvt", "msn", "mst"
# ncov     : struttura Σ (1 = comune, 2 = diag, 3 = libera, 4/5 altre varianti)
# itmax    : iterazioni EM massime
# epsilon  : soglia di convergenza sul log‑lik
#
# Valore di ritorno
# -----------------
# lista con elementi:
#   • distr, error, loglik, bic, aic
#   • pro, mu, sigma, dof, delta
#   • clust (MAP), tau (posteriori n×g), lk (traccia log‑lik)
################################################################################

EmSkewfit2 <- function(dat, g, init,
                       distr, ncov,
                       itmax, epsilon)
{
  ##########################################################################
  # 0) Controlli preliminari su modello e dati
  ##########################################################################
  ndist <- switch(tolower(distr),
                  "mvn" = 1,  # Normal
                  "mvt" = 2,  # t
                  "msn" = 3,  # Skew‑Normal
                  "mst" = 4,  # Skew‑t
                  5)          # → valore sentinella per *not available*
  
  # Modelli non implementati o ncov fuori range
  if (ndist > 4 || ncov < 1 || ncov > 5)
    stop("the model specified is not available yet")
  
  # Trasformiamo i dati in matrice pura (più sicuro per .C)
  dat <- as.matrix(dat)
  n   <- nrow(dat)
  p   <- ncol(dat)
  
  browser()
  # I modelli di mistura hanno bisogno di “abbastanza” osservazioni:
  # regola euristica n ≥ 20*g (vale per stime stabili di Σ_k e δ_k)
  if (n <= 20 * g)
    stop("sample size is too small")
  
  # Deve esserci init non‑NULL; se assente l’utente ha sbagliato workflow
  if (is.null(init) || missing(init))
    stop("init should be provided")
  
  ##########################################################################
  # 1) Un‑pack dei parametri iniziali
  ##########################################################################
  pro   <- init$pro            # (g)           mixing proportions
  mu    <- init$mu             # (p × g)       medie
  sigma <- init$sigma          # (p × p × g)   covarianze
  dof   <- init$dof            # (g)           ν_k (solo se t / skew‑t)
  delta <- init$delta          # (p × g)       vettori skew

    
  ##########################################################################
  # 2) Chiamata alla routine C – EM principale
  #    Vengono passati i parametri iniziali *by value* (copiati) per non
  #    modificarli in‑place e poter ispezionare a posteriori; la routine li
  #    sovrascrive con gli stimatori finali.
  ##########################################################################
  obj <- .C(
    'emskewfit2',                      #  1  simbolo C
    PACKAGE = "EMMIXskew",             #      pacchetto sorgente
    
    # --- Input “costanti” (non modificati) -------------------------------
    as.double(dat),                    #  2  dati (column‑major)
    as.integer(n),                     #  3  n
    as.integer(p),                     #  4  p
    as.integer(g),                     #  5  # componenti
    as.integer(ncov),                  #  6  struttura Σ
    as.integer(ndist),                 #  7  codice distribuzione
    
    # --- Parametri che verranno aggiornati dalla C‑routine --------------
    pro   = as.double(pro),            #  8  π_k
    mu    = as.double(mu),             #  9  μ_k
    sigma = as.double(sigma),          # 10  Σ_k
    dof   = as.double(dof),            # 11  ν_k
    delta = as.double(delta),          # 12  δ_k
    
    # --- Buffer per l’E‑step --------------------------------------------
    tau   = double(n * g),             # 13  posteriori τ_ik
    double(n * g),                     # 14  v_ik
    double(n * g),                     # 15  z_ik
    double(n * g),                     # 16  log‑densità
    double(n * g),                     # 17  placeholder
    
    # --- Quantità aggregate ---------------------------------------------
    sumtau = double(g),                # 18  Σ_i τ_ik
    sumvt  = double(g),                # 19  Σ_i τ_ik v_ik
    sumzt  = double(g),                # 20  Σ_i τ_ik z_ik
    sumlnv = double(g),                # 21  Σ_i τ_ik ln(v_ik)
    
    # --- Traccia log‑lik e criteri informativi --------------------------
    loglik = double(1),                # 22  log‑lik finale
    lk     = double(itmax),            # 23  log‑lik ad ogni iterazione
    aic    = double(1),                # 24  AIC
    bic    = double(1),                # 25  BIC
    
    # --- Output classificazione e codice errore -------------------------
    clust = integer(n),                # 26  cluster MAP
    error = integer(1),                # 27  codice errore (0 = ok)
    
    # --- Parametri di controllo -----------------------------------------
    as.integer(itmax),                 # 28  max iterazioni
    as.double(epsilon)                 # 29  soglia convergenza
  )[c(7:12, 21:26)]                    # ← selezioniamo soltanto i campi utili
  
  ##########################################################################
  # 3) Post‑processing dell’output
  ##########################################################################
  lk <- obj$lk
  lk <- lk[lk != 0]  # la routine pre‑alloca `lk`; rimuoviamo tail inutilizzato

    
  # Costruiamo la lista di ritorno (in ogni caso, anche se error > 0)
  list(
    distr = distr,           # tipo di distribuzione
    error = obj$error,       # 0 = convergenza
    loglik = obj$loglik,     # log‑lik finale
    bic   = obj$bic,         # BIC
    aic   = obj$aic,         # AIC
    
    # --- Parametri stimati ------------------------------------------------
    pro   = obj$pro,                           # (g)
    mu    = array(obj$mu,    c(p, g)),         # (p × g)
    sigma = array(obj$sigma, c(p, p, g)),      # (p × p × g)
    dof   = obj$dof,                           # (g) (NA per Normal/Skew‑N)
    delta = array(obj$delta, c(p, g)),         # (p × g)
    
    # --- Output clustering & traccia --------------------------------------
    clust = obj$clust,                        # cluster MAP su ogni obs.
    tau   = array(obj$tau, c(n, g)),          # posteriori τ (n × g)
    lk    = lk                                # vettore log‑lik iterativo
  )
}






################################################################################
################################################################################
################################################################################
################################################################################

# 5)

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
# getICL
# ------------------------------------------------------------------------------
# Calcola il **criterio ICL** (Integrated Completed‑data Likelihood) per un
# modello di miscela stimato, a partire da:
#   • dati grezzi  x               (n × p)
#   • parametri    pro, mu, sigma, dof, delta
#   • assegnazioni hard  clust     (lunghezza n, valori 1…g)
#
# ICL = log P(x, z | θ̂)  –  (q/2) log n
#      = log‑verosimiglianza *completa* − penalità dimensionale
# dove q = # parametri liberi del modello.
#
# Il termine di verosimiglianza “completa” include le etichette z (cluster)
# ed è quindi diverso dalla log‑lik usata per BIC.
################################################################################

getICL <- function(x, n, p, g,
                   distr, ncov,
                   pro, mu, sigma, dof, delta,
                   clust)
{
  # --------------------------------------------------------------------------
  # 0) Preparazione e variabili di accumulo
  # --------------------------------------------------------------------------
  x      <- as.matrix(x)   # garantiamo matrice
  loglik <- 0              # log‑lik completa
  nc     <- 0              # # parametri liberi (da calcolare più sotto)
  browser()
  
  # --------------------------------------------------------------------------
  # 1) Numero efficace di osservazioni (esclude eventuali outlier codificati 0)
  # --------------------------------------------------------------------------
  ## Se si usa la convenzione che gli outlier/ruis sono etichettati 0,
  ## li escludiamo da n nella penalità. (nn = n effettivo)
  nn <- sum(clust > 0)
  
  # --------------------------------------------------------------------------
  # 2) Valutazione della densità log del modello per **ogni** componente
  #    ddmix restituisce matrice n × g di densità (o log‑densità) condizionate.
  # --------------------------------------------------------------------------
  lnden <- as.matrix(
    ddmix(x, n, p, g,
          distr,
          mu, sigma, dof, delta)     # routine C `ddmix2` sotto il cofano
  )
  
  # --------------------------------------------------------------------------
  # 3) Log‑lik “completa”: sommiamo, per ogni osservazione,
  #    log π_h  +  log f_h(x_i)  soltanto sul cluster assegnato (clust == h)
  # --------------------------------------------------------------------------
  for (h in 1:g)
    loglik <- loglik +
    sum( ifelse(clust == h,
                log(pro[h]) + lnden[, h],   # contributo se i ∈ h
                0) )                        # altrimenti 0

    
  # --------------------------------------------------------------------------
  # 4) Conteggio # parametri liberi (q) in funzione di:
  #      • struttura Σ (ncov)
  #      • distribuzione base (distr)
  #
  #    pro  : g‑1  (vincolo Σπ_k =1)
  #    mu   : g * p
  #    sigma: dipende da ncov
  # --------------------------------------------------------------------------
  nc <- switch(tolower(ncov),
               # --- Covarianza “comune” a tutte le componenti -------------------------
               '1' = (g - 1) +          # mixing proportions
                 g * p +            # medie
                 p * (1 + p) / 2,   # unica Σ condivisa
               
               # --- Cov. comune ma diagonale -----------------------------------------
               '2' = (g - 1) +          # pro
                 g * p +            # mu
                 p,                 # varianze diagonali comuni
               
               # --- Cov. generali (una per cluster) ----------------------------------
               '3' = (g - 1) +
                 g * ( p + p * (1 + p) / 2 ),  # g medie + g Σ_k completi
               
               # --- Cov. diagonali indipendenti per cluster --------------------------
               '4' = (g - 1) +
                 g * ( p + p ),     # g medie + g varianze diag (p) +? (qui doppio p)
               
               # --- Sferiche distinte (σ_h * I_p) ------------------------------------
               '5' = (g - 1) +
                 g * ( p + 1 )      # g medie + g scalar var
  )
  
  # --------------------------------------------------------------------------
  # 5) Parametri aggiuntivi specifici della distribuzione
  # --------------------------------------------------------------------------
  nc <- switch(tolower(distr),
               "mvn" = nc,                 # nessun parametro aggiuntivo
               "mvt" = nc + g,             # +1 ν_k per componente
               "msn" = nc + g * p,         # +p δ_k per comp.
               "mst" = nc + g * p + g      # +p δ_k  + ν_k
  )

    
  # --------------------------------------------------------------------------
  # 6) Calcolo ICL: log‑lik completa – (q/2) log n_eff
  # --------------------------------------------------------------------------
  ICL <- loglik - (nc / 2) * log(nn)
  
  # --------------------------------------------------------------------------
  # 7) Ritorna in formato lista (compatibile con uso altrove)
  # --------------------------------------------------------------------------
  list(ICL = ICL)
}



################################################################################
################################################################################
################################################################################
################################################################################

# 6 && 9)

################################################################################
################################################################################
################################################################################
################################################################################





################################################################################
# ddmix
# ------------------------------------------------------------------------------
# Calcola (log‑)densità per ciascuna osservazione e ciascuna componente di
# una miscela a distribuzioni multivariate „skew” (Normal, t, Skew‑Normal,
# Skew‑t).  Funzione di *servizio* essenziale per:
#   • computing ICL / BIC / AIC fuori dall’EM,
#   • valutare densità in punti di griglia,
#   • calcoli post‑hoc (ad es. contour plot).
#
# Tutto il lavoro numericamente intenso viene svolto nella routine C
# `ddmix2` (parte del package **EMMIXskew**), scritta per sfruttare BLAS
# e funzioni speciali (CDF t multivariata troncata).
#
# Argomenti
# ---------
# dat   : matrice n × p dei dati
# n, p  : scorciatoie per nrow(dat) e ncol(dat) (riduce ricontrolli)
# g     : numero di componenti di miscela
# distr : "mvn" | "mvt" | "msn" | "mst"
# mu    : array p × g di medie
# sigma : array p × p × g di matrici di covarianza
# dof   : vettore g di gradi di libertà (usato se distribuzione t/skew‑t)
# delta : array p × g di parametri di asimmetria (usato se distribuzione skew)
#
# Valore di ritorno
# -----------------
# matrice n × g contenente, in posizione [i, k], la log‑densità (o densità)
# della k‑esima componente valutata sull’i‑esima osservazione.
################################################################################

ddmix <- function(dat, n, p, g,
                  distr,
                  mu, sigma,
                  dof   = NULL,
                  delta = NULL)
{
  ## -------------------------------------------------------------------------
  ## 0) Default per parametri facoltativi
  ## -------------------------------------------------------------------------
  if (is.null(dof))
    dof <- rep(4, g)                 # convenzione: ν = 4 se non fornito
  
  if (is.null(delta))
    delta <- array(0, c(p, g))       # nessuna asimmetria ⇒ δ = 0
  
  ## -------------------------------------------------------------------------
  ## 1) Mappatura della distribuzione su intero (coerente con codice C)
  ## -------------------------------------------------------------------------
  ndist <- switch(tolower(distr),
                  "mvn" = 1,
                  "mvt" = 2,
                  "msn" = 3,
                  "mst" = 4,
                  5)                  # 5 = modello non implementato (errore)
  browser()
  
  if (ndist > 4)
    stop("the model specified is not available yet")
  
  ## -------------------------------------------------------------------------
  ## 2) Verifiche di consistenza sui dati
  ## -------------------------------------------------------------------------
  dat <- as.matrix(dat)              # assicura struttura matrice
  
  # Se dat è 1 × 1 trasformiamo in (1, 1) correttamente
  if (n == 1 && (ncol(dat) == 1))
    dat <- t(dat)
  
  if (nrow(dat) != n || ncol(dat) != p)
    stop("dat does not match n and p.")
  
  ## -------------------------------------------------------------------------
  ## 3) Verifiche sui parametri di input (dimensioni/shape)
  ## -------------------------------------------------------------------------
  if (length(c(mu)) != (p * g))
    stop(paste("mu should be a", p, 'by', g, "matrix!"))
  
  if (length(c(sigma)) != (p * p * g))
    stop(paste("sigma should be a", p, 'by', p, 'by', g, "array!"))
  
  if (length(c(dof)) != g)
    stop(paste("dof should be a", g, "vector!"))
  
  if (length(c(delta)) != (p * g))
    stop(paste("delta should be a", p, 'by', g, "array!"))

    
  ## -------------------------------------------------------------------------
  ## 4) Chiamata alla routine C per il calcolo delle densità
  ## -------------------------------------------------------------------------
  obj <- .C(
    'ddmix2',                       # routine C ottimizzata
    PACKAGE = "EMMIXskew",
    
    # --- input principali ---------------------------------------------------
    as.double(dat),                 # 1  dati flatten col‑major
    as.integer(n),                  # 2  # osservazioni
    as.integer(p),                  # 3  dimensione
    as.integer(g),                  # 4  # componenti
    as.integer(ndist),              # 5  codice distribuzione
    as.double(mu),                  # 6  μ
    as.double(sigma),               # 7  Σ
    as.double(dof),                 # 8  ν
    as.double(delta),               # 9  δ
    
    # --- output -------------------------------------------------------------
    den   = double(n * g),          # 10 densità/log‑dens flatten
    error = integer(1)              # 11 codice errore
  )[10:11]                          # estraiamo solo 'den' e 'error'
  
  ## -------------------------------------------------------------------------
  ## 5) Gestione eventuale errore dalla routine C
  ## -------------------------------------------------------------------------
  if (obj$error)
    stop("error")
  
  ## -------------------------------------------------------------------------
  ## 6) Ritorna matrice n × g (reshape di `den`)
  ## -------------------------------------------------------------------------
  matrix(obj$den, ncol = g)
}



################################################################################
################################################################################
################################################################################
################################################################################

# 7)

################################################################################
################################################################################
################################################################################
################################################################################





################################################################################
# EmSkewMOD
# ------------------------------------------------------------------------------
# Scopo  : Individuare un “mode point” (massimo di densità) per ciascuna
#          componente di una miscela multivariata.  Per i modelli simmetrici
#          (MVN / MVT) il mode coincide con la media μ, mentre per i modelli
#          *skew* (MSN / MST) non è noto in forma chiusa: viene quindi stimato
#          per simulazione Monte‑Carlo.
#
# Strategia (componenti skew):
#   1. Campioniamo `nrand` osservazioni dalla k‑esima distribuzione con
#      funzione ausiliaria `mvrand()`.
#   2. Valutiamo la densità (log) in tutti i punti campionati tramite `ddmix`.
#   3. Selezioniamo il punto a densità massima e lo salviamo come mode.
#
# Argomenti
# ---------
# p      : dimensione variabile multivariata
# g      : numero di componenti di miscela
# distr  : stringa "mvn" | "mvt" | "msn" | "mst"
# mu     : matrici p × g di medie
# sigma  : array  p × p × g di covarianze
# dof    : vettore g di gradi di libertà (solo t / skew‑t)
# delta  : matrici p × g di parametri di asimmetria (solo skew‑*)
# nrand  : n° campioni Monte‑Carlo per ricerca del massimo (default 10 000)
#
# Ritorno
# -------
# Matrice p × g con i “mode points” stimati per ciascun componente.
################################################################################

EmSkewMOD <- function(p, g,
                      distr,
                      mu, sigma,
                      dof, delta,
                      nrand = 10000)
{
  distr <- tolower(distr)  # normalizza in minuscolo per robustezza
  browser()
  
  ## -------------------------------------------------------------------------
  ## 1) Funzione ausiliaria: genera campioni da MSN o MST
  ##    (per MVN/MVT non viene mai invocata)
  ## -------------------------------------------------------------------------
  mvrand <- function(n, p, distr, mean, cov, nu, del)
  {
    switch(distr,
           # `rdmsn` e `rdmst` sono generatori presenti in EMMIXskew
           'msn' = rdmsn(n, p, mean = mean, cov = cov,       del = del),
           'mst' = rdmst(n, p, mean = mean, cov = cov, nu = nu, del = del),
           NULL)  # per sicurezza, anche se non dovremmo mai arrivare qui
  }
  
  ## -------------------------------------------------------------------------
  ## 2) Verifica che il modello richiesto sia supportato
  ## -------------------------------------------------------------------------
  if (!(distr %in% c("mvn", "mvt", "msn", "mst")))
    stop("model specified not available yet")

  ## -------------------------------------------------------------------------
  ## 3) Rimodellamento parametri in array/matrici di forma coerente
  ##    (vengono passati come vettori flatten dai livelli inferiori)
  ## -------------------------------------------------------------------------
  mu    <- array(mu,    c(p, g))      # medie  (p × g)
  sigma <- array(sigma, c(p, p, g))   # Σ_k    (p × p × g)
  dof   <- c(dof)                     # ν_k    (g)
  delta <- array(delta, c(p, g))      # δ_k    (p × g)
  
  ## -------------------------------------------------------------------------
  ## 4) Se il modello è simmetrico, il mode coincide con μ
  ## -------------------------------------------------------------------------
  modpts <- mu                        # matrice risultato inizializzata
  if (distr %in% c("mvn", "mvt"))
    return(modpts)

  ## -------------------------------------------------------------------------
  ## 5) Caso skew: ricerca Monte‑Carlo componente per componente
  ## -------------------------------------------------------------------------
  for (h in 1:g) {
    
    ## 5a) Campionamento pseudo‑random dalla k‑esima distribuzione
    dat <- cbind(
      mvrand(nrand, p, distr,
             mu[, h],            # μ_k
             sigma[, , h],       # Σ_k
             dof[h],             # ν_k
             delta[, h])         # δ_k
    )
    
    ## 5b) Valutazione della densità (log) per tutti i campioni generati
    den <- ddmix(dat, nrand, p, 1,      # g = 1 perché valutiamo una sola comp.
                 distr,
                 mu[, h],
                 sigma[, , h],
                 dof[h],
                 delta[, h])
    
    ## 5c) Selezione del punto a massima densità
    id <- which.max(den)
    modpts[, h] <- dat[id, ]
  }
  
  ## -------------------------------------------------------------------------
  ## 6) Restituisce la matrice (p × g) dei mode points
  ## -------------------------------------------------------------------------
  return(modpts)
}

################################################################################
################################################################################
################################################################################
################################################################################

# 8)

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
# rdmsn
# ------------------------------------------------------------------------------
# Generatore di campioni dalla **Distribuzione Skew‑Normale Multivariata** di
# Sahu, Dey & Branco (2003) – versione non ristretta.
#
# Rappresentazione stocastica utilizzata:
#   Y = μ + X + |Z| δ
# con
#   • X  ~  N_p(0, Σ)           (vettore multivariato normale)
#   • Z  ~  N(0, 1)             (scalare)  →   |Z| introduce l’asimmetria
#   • δ  ∈ ℝ^p                  (vettore di skew)
#
# In altre parole, ad un normale multivariato aggiungiamo uno “shift” positivo
# lungo la direzione δ, di ampiezza |Z|; ciò inclina la distribuzione senza
# alterare la matrice di covarianza di base.
#
# Argomenti
# ---------
# n      : numero di osservazioni da simulare
# p      : dimensione dello spazio (numero variabili)
# mean   : vettore μ di lunghezza p (default 0)
# cov    : matrice di covarianza Σ (p × p) (default I_p)
# del    : vettore δ di skew (lunghezza p; default 0 → normale simmetrica)
#
# Valore di ritorno
# -----------------
# matrice n × p di campioni simulati.
################################################################################
rdmsn <- function(n, p,
                  mean = rep(0, p),
                  cov  = diag(p),
                  del  = rep(0, p))
{
  browser()
  x <- rdmvn(n, p, mean, cov)     # passo 1: X  ~  N_p(μ, Σ)
  z <- abs(rnorm(n))              # passo 2: |Z| ~ |N(0,1)|
  as.matrix( z %*% t(del) + x )   # Y = X + |Z| δ   (ritorna matrice n×p)
}



################################################################################
# rdmst
# ------------------------------------------------------------------------------
# Generatore di campioni dalla **Distribuzione Skew‑t Multivariata** di
# Sahu, Dey & Branco (2003) – versione non ristretta.
#
# Rappresentazione stocastica (hierarchical):
#   U      ~  Gamma(ν/2, ν/2)               (scala di “weight”)
#   X | U  ~  N_p(0, Σ / U)                 (normale con varianza scalata)
#   Z | U  ~  N(0, 1 / U)                   (scalare indipendente da X)
#   Y      = μ + X + |Z| δ                  (shift skew)
#
# Ciò produce una distribuzione con:
#   • Code pesanti controllate da ν         (quando ν → ∞ si ottiene la skew‑N)
#   • Asimmetria introdotta da δ
#
# Argomenti
# ---------
# n      : numero di campioni
# p      : dimensione
# mean   : vettore μ
# cov    : matrice Σ
# nu     : gradi di libertà ν (>0), default 10
# del    : vettore δ di skew
#
# Valore di ritorno
# -----------------
# matrice n × p di osservazioni simulate.
################################################################################
rdmst <- function(n, p,
                  mean = rep(0, p),
                  cov  = diag(p),
                  nu   = 10,
                  del  = rep(0, p))
{
  broswer()
  # passo 1: variabile di scala U  ~  Gamma(ν/2, ν/2)
  u <- rgamma(n, nu / 2, nu / 2)
  
  # passo 2: X | U  ~  N_p(μ, Σ / U)
  x <- t( t( rdmvn(n, p, cov = cov) / sqrt(u) ) + mean )
  
  # passo 3: Z | U  ~  |N(0, 1 / U)|
  z <- abs( rnorm(n) / sqrt(u) )
  
  # passo 4: restituisce Y = X + |Z| δ
  as.matrix( z %*% t(del) + x )
}

# 
# 
# 
# EmSkewfit1<-function(dat,g,clust,distr,ncov,itmax,epsilon,initloop=20)
# {
# ndist<-switch(tolower(distr),"mvn"=1,"mvt"=2,"msn"=3,"mst"=4,5)
# if(ndist>4|ncov<1|ncov>5) 
# stop("the model specified is not available yet")
# 
# dat<-as.matrix(dat)
# n <- nrow(dat)
# p <- ncol(dat)
# 
# if(n <= 20*g)
# stop("sample size is too small")
# 
# if(missing(clust) | is.null(clust))
# stop("initial clust must be given")
# 
# clust <- unclass(as.ordered(clust))
# 
# if(max(clust)!=g) stop(paste("The levels of cluster should be g=",g))
# 
# 
# obj<-.C('emskewfit1',PACKAGE="EMMIXskew",
# as.double(dat),as.integer(n),as.integer(p),
# as.integer(g),as.integer(ncov),as.integer(ndist),#6
# pro   = double(g),mu  = double(p*g),sigma  = double(p*p*g),
# dof   = double(g),delta=double(p*g), #11
# tau   = double(n*g),double(n*g),double(n*g),double(n*g),double(n*g),
# sumtau=double(g), sumvt=double(g),
# sumzt=double(g), sumlnv=double(g), 
# ewy=double(p*g),ewz=double(p*g),ewyy = double(p*p*g),#23
# loglik= double(1),lk= double(itmax),aic= double(1),bic= double(1),
# clust = as.integer(clust),#28
# error = integer(1),as.integer(itmax),
# as.double(epsilon),as.integer(initloop))[c(7:12,24:29)]
# 
# lk<-obj$lk;lk<-lk[lk!=0]
# 
# list(distr=distr,error=obj$error,loglik=obj$loglik,bic=obj$bic,aic=obj$aic,
# pro=obj$pro,mu= array(obj$mu,c(p,g)),
# sigma=array(obj$sigma,c(p,p,g)),dof=obj$dof,delta=array(obj$delta,c(p,g)),
# clust=obj$clust,tau=array(obj$tau,c(n,g)),lk=lk)
# }
# 
# 
# 
# # distance functions
# 
# 
# # calculate the Scale free Weighted (mahalonobis distance) Ratio (SWR)and UWR
# getSWR<-function(dat,g,sigma,clust, tau)
# {
# 
# ret <- NULL
# 
# if(g>1)
# {
# 
# intra <- intradist(dat,g,sigma, clust, tau) 
# if(intra$error) stop("error")
# 
# inter <- interdist(dat,g,sigma, clust, tau) 
# if(inter$error) stop("error")
# 
# ret <- list(SWR=sqrt(intra$OverallIntraDist1/inter$OverallInterDist1),
# UWR=sqrt(intra$OverallIntraDist2/inter$OverallInterDist2))
# }
# 
# ret
# 
# }
# 
# 
# 
# mypanel2<- function(x,y,...) {
# par(new=TRUE);
# smoothScatter(x,y,..., nrpoints=0)
# }
# 
# mypanel3<- function(x,y,...) {
# par(new=TRUE);
# Lab.palette <- colorRampPalette(c("blue", "orange", "red"), space = "Lab")
# smoothScatter(x,y, colramp = Lab.palette)
# 
# }
# 
# mypanel4<- function(x,y,...) {
# xy <- cbind(x,y)
# par(new=TRUE);
# plot(xy, col = densCols(xy), pch=20)
# }
# 
# panel.density <- function(x, col=1,...)
# {
# usr <- par("usr"); on.exit(par(usr))
# par(usr = c(usr[1:2], 0, 1.5) )
# 
# oo = density(x)
# y = oo$y
# lines(oo$x,y/max(y))
# 
# }
# 
# conplot <-function(x,y, pro, mu, sigma, dof, delta,distr,
# grid=300, nrand=6000,levels=seq(5,95,by=20),col ='white')
# {
# 
# ndist<-switch(tolower(distr),"mvn"=1,"mvt"=2,"msn"=3,"mst"=4,5)
# if(ndist>4) 
# stop("the model specified is not available yet")  
# 
# ddemmix <- function(dat, n, p, g, distr, pro, mu, sigma, dof, delta)
# {
# ret<-ddmix(dat,n,p,g, distr, mu,sigma,dof,delta)
# c(exp(ret)%*%pro )
# } #joint density
# 
#     xlim = range(x)+c(-1,0)
#     ylim = range(y)+c(-1,0)
# 
# g<-length(pro);p<-2
# 
# x1 <- seq(xlim[1], xlim[2], length=grid) 
# y1 <- seq(ylim[1], ylim[2], length=grid) 
# 
# nx <- length(x1)
# ny <- length(y1)
# xoy <- cbind(rep(x1,ny), as.vector(matrix(y1,nx,ny,byrow=TRUE)))
# X <- matrix(xoy, nx*ny, 2, byrow=FALSE)
# 
# dens     <- ddemmix(X, nx*ny,2,g, distr, pro, mu, sigma, dof, delta)
# dens.mat <- matrix(dens,nx,ny)
# 
# n <- table(sample(1:g, nrand, replace = TRUE, prob = pro))
# nn <- n
# if(length(n) <g) {
# nn <- rep(0,g)
# for(i in as.numeric(names(n)))
# nn[i] <- n[paste(i)]
# }
# rand     <-  rdemmix(nn,p,g,distr,mu,sigma,dof,delta)
# rand.den <-  ddemmix(rand, nrand,2,g, distr, pro, mu, sigma, dof, delta)
# cont     <-  quantile(rand.den, prob=levels/100)
# contour(x1, y1, dens.mat, levels=cont, add=TRUE,drawlabels=FALSE,lty=1,col =col)
# }
# 
# conplot2 <- function (x, y, pro, mu, sigma, dof, delta, distr, grid = 300, 
#     nrand = 6000, levels = seq(5, 95, by = 20)) 
# {
#     ndist <- switch(tolower(distr), mvn = 1, mvt = 2, msn = 3, mst = 4, 
#         5)
#     if (ndist > 4) 
#         stop("the model specified is not available yet")
#     ddemmix <- function(dat, n, p, g, distr, pro, mu, sigma, 
#         dof, delta) {
#         ret <- ddmix(dat, n, p, g, distr, mu, sigma, dof, delta)
#         c(exp(ret) %*% pro)
#     }
#     g <- length(pro)
#     p <- 2
# 
# #make mesh
# 
#     xlim = range(x)+c(-1,0)
#     ylim = range(y)+c(-1,0)
# 
#     x1 <- seq(xlim[1], xlim[2], length= grid)
#     y1 <- seq(ylim[1], ylim[2], length= grid)
# 
#     nx <- length(x1)
#     ny <- length(y1)
# 
#     xoy <- cbind(rep(x1, ny), as.vector(matrix(y1, nx, ny, byrow = TRUE)))
#     X <- matrix(xoy, nx * ny, 2, byrow = FALSE)
#     dens <- ddemmix(X, nx * ny, p, g, distr, pro, mu, sigma, 
#         dof, delta)
# 
#     dens.mat <- matrix(dens, nx, ny)
# 
# #
#     n <- table(sample(1:g, nrand, replace = TRUE, prob = pro))
#     nn <- n
#     if (length(n) < g) {
#         nn <- rep(0, g)
#         for (i in as.numeric(names(n))) nn[i] <- n[paste(i)]
#     }
# 
#     rand <- rdemmix(nn, p, g, distr, mu, sigma, dof, delta)
#     rand.den <- ddemmix(rand, nrand, 2, g, distr, pro, mu, sigma,dof, delta)
#     cont <- quantile(rand.den, prob = 1-levels/100)
# 
#     samp <- cbind(x,y)
#     samp.den <-ddemmix(samp, length(x), 2, g, distr, pro, mu, sigma,dof, delta)
# 
# select <-  which(samp.den>cont)
# 
# clust  <-  ifelse(samp.den>cont,2,1)
# 
# list(select,clust,x1=x1, y1=y1, dens.mat=dens.mat, cont=cont)
# }
# 
# 
# conplot3 <- function (x, y, pro, mu, sigma, dof, delta, modpts,distr, grid =300, 
#     nrand = 10000, levels = seq(5, 95, by = 20)) 
# {
#     ndist <- switch(tolower(distr), mvn = 1, mvt = 2, msn = 3, mst = 4, 
#         5)
#     if (ndist > 4) 
#         stop("the model specified is not available yet")
#     ddemmix <- function(dat, n, p, g, distr, pro, mu, sigma, 
#         dof, delta) {
#         ret <- ddmix(dat, n, p, g, distr, mu, sigma, dof, delta)
#         c(exp(ret) %*% pro)
#     }
# 
#     g <- length(pro)
#     p <- 2
# 
# #----------------------------------------------------
# #mesh
# #----------------------------------------------------
# 
# #make mesh
# 
#     xlim = range(x)+c(-1,0)
#     ylim = range(y)+c(-1,0)
# 
#     x1 <- seq(xlim[1], xlim[2], length=grid)
#     y1 <- seq(ylim[1], ylim[2], length=grid)
# 
#     nx <- length(x1)
#     ny <- length(y1)
# 
#     xoy <- cbind(rep(x1, ny), as.vector(matrix(y1, nx, ny, byrow = TRUE)))
# 
#     X <- matrix(xoy, nx * ny, 2, byrow = FALSE)
# 
# #--------------------------------------------------
# 
# 
# 
# for(h in 1:g) { #do each component one by one
# 
#     dens <- ddemmix(X, nx * ny, 2, 1, distr, c(1), mu[,h], sigma[,,h],dof[h], delta[,h])
# 
#     dens.mat <- matrix(dens, nx, ny)
# 
# #  randon sample
# 
#     rand <- rdemmix(c(nrand), 2, 1, distr, mu[,h], sigma[,,h], dof[h], delta[,h])
# 
#     rand.den <- ddemmix(rand, nrand, 2, 1, distr, c(1), mu[,h], sigma[,,h], dof[h], delta[,h])
# 
# #-----------------------------------------
# 
#     cont <- quantile(rand.den, prob = 1-levels/100)
# 
#     contour(x1, y1, dens.mat, levels = cont, add = TRUE, drawlabels = FALSE,lty = 1, col = h)
#     
#         if (!is.null(modpts)) 
#         points(t(modpts[, h]), col = h,pch=3)
# } #end of h loop
# 
# 
# }
# 
# EmSkew.filter <- function (S, g=1,distr="mst", diag.panel = TRUE, upper.panel = "type2", 
#     lower.panel = "type3", levels = 90, attop = FALSE,title="",path="",plot=TRUE) 
# {
#     
# S <- as.matrix(S)
# 
# dat <- S[,1:2]
# 
# obj <- EmSkew(dat,g,distr,ncov=3,itmax=200,debug=0)
# 
# ppp <- conplot2(c(dat[,1]), c(dat[,2]), obj$pro, obj$mu, obj$sigma, 
# obj$dof, obj$delta, obj$distr, nrand=10000,levels = levels)
# 
# clust <- ppp[[2]]
# select<- ppp[[1]]
# 
# #---------------------------------
# 
# mypanel <- function(x, y, ...) {
# 
#         par(new = TRUE)
#         points(x, y, ..., col = clust)
# 
#         st <- pmatch(c(x[1], y[1]), S[1, ])
# #st <- c(current.row(),current.column())
# 
# if(st[1]==1&st[2]==2) {
# 
# a=contourLines(ppp$x1, ppp$y1, ppp$dens.mat, levels=ppp$cont)[[1]]
# 
# ax <- a$x
# #ax[ax<ppp$x1[1]]=ppp$x1[1]+1
# 
# ay <- a$y
# 
# lines(ax,ay,lty = 2, col = 'blue')
# 
# }
# 
# }
# 
#     if (diag.panel) {
#         diag.panel <- panel.density
#     }
#     else diag.panel <- NULL
# 
# 
#     upper.panel <- switch(upper.panel, type2 = mypanel2, type3 = mypanel3, 
#         type4 = mypanel4, NULL)
# 
#     lower.panel <- switch(lower.panel, type2 = mypanel2, type3 = mypanel3, 
#         type4 = mypanel4, NULL)
# 
# #-------------------------------
# 
# if(plot) {
# 
# dev.new()
# pairs(S, pch = ".", panel=mypanel,row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,main=paste("Before Filtering (EmSkew):",toupper(distr)))
# 
# 
# dev.new()
# pairs(S[select,], upper.panel=upper.panel,row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,main=paste("After Filtering (EmSkew):",toupper(distr)))
# }
# 
# #-------------------------------
# 
# if(path!='') {
# 
# png(paste(path,'/',title,"-before.png",sep=''),width=512,height=512)
#  
# pairs(S, pch = ".", panel=mypanel,row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,main=paste("Before Filtering (EmSkew):",toupper(distr)))
# 
# dev.off()
# 
# png(paste(path,'/',title,"-after.png",sep=''),width=512,height=512)
#  
# pairs(S[select,], upper.panel=upper.panel,row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,main=paste("After Filtering (EmSkew):",toupper(distr)))
# dev.off()
# 
# }
# 
# list(subset=select,clust=clust,filter=obj)
# }
# 
# 
# 
# 
# #------------------------------------------------------------------------------#
# EmSkew.contours <- function (S, obj = NULL, clust = NULL,distr="",diag.panel = TRUE, upper.panel = "type2", 
#     lower.panel = "type3", levels = seq(5, 95, by = 20), plot=TRUE, title="",path='',attop = FALSE) 
# {
#     mypanel <- function(x, y, ...) {
#         par(new = TRUE)
#         smoothScatter(x, y, ..., nrpoints = 0)
#         g <- length(obj$pro)
#         st <- pmatch(c(x[1], y[1]), S[1, ])
# #st <- c(current.row(),current.column())
# 
# 	
# 	conplot3(x, y, obj$pro, obj$mu[st, ], obj$sigma[st, st, 
#             ], obj$dof, obj$delta[st, ],obj$modpts[st,], obj$distr, levels = levels)
#     }
#     if (diag.panel) {
#         diag.panel <- panel.density
#     }
#     else diag.panel <- NULL
#     upper.panel <- switch(upper.panel, type2 = mypanel2, type3 = mypanel3, 
#         type4 = mypanel4, NULL)
#     lower.panel <- switch(lower.panel, type2 = mypanel2, type3 = mypanel3, 
#         type4 = mypanel4, NULL)
# 
# 
# if(plot) {
# 
#     if (is.null(clust)) {
#         if (!is.null(obj)) {
#             pairs(S, panel = mypanel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("Contours of Components using EmSkew:",toupper(obj$distr), "Distribution") )
#         }
#         else {
#             pairs(S, upper.panel = upper.panel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("The heatmap pairwise plots of the data"))
#         }
#     }
#     else pairs(S, pch = ".", col = clust, row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,
# 	main=paste("Clustering using EmSkew:",toupper(distr), "Distribution") )
# 
# }
# 
# 
# if(path!='') {
# 
# png(paste(path,'/',title,".png",sep=''),width=512,height=512)
# 
#     if (is.null(clust)) {
#         if (!is.null(obj)) {
#             pairs(S, panel = mypanel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("Contours of Components using EmSkew:",toupper(obj$distr), "Distribution") )
#         }
#         else {
#             pairs(S, upper.panel = upper.panel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("The heatmap pairwise plots of the data"))
#         }
#     }
#     else pairs(S, pch = ".", col = clust, row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,
# 	main=paste("Clustering using EmSkew:",toupper(distr), "Distribution") )
# 
# dev.off()
# 
# }
# 
# }
# #------------------------------------------------------------------------------#
# 
# 
# EmSkew.flow <-function(S,obj=NULL,distr="",diag.panel=TRUE,
# upper.panel="type2",lower.panel="type3",
# levels=seq(5,95,by=20),attop=FALSE,clust=NULL,title="",path="",plot=TRUE) {
# 
# mypanel<- function(x,y,...) {
# par(new=TRUE);
# smoothScatter(x,y,..., nrpoints=0)
# g <- length(obj$pro)
# 
# st <- pmatch(c(x[1],y[1]),S[1,])
# #st <- c(current.row(),current.column())
# 
# 
# conplot(x,y, obj$pro,obj$mu[st,],obj$sigma[st,st,],
# obj$dof,obj$delta[st,],obj$distr,levels=levels,nrand=10000)
# 
# if(!is.null(obj$modpts))
# points(t(obj$modpts[st,]),col=1:g,pch=3)
# }
# 
# 
# 
# if(diag.panel) {
# diag.panel<- panel.density}
# else diag.panel<- NULL
# 
# upper.panel<-switch(upper.panel,
#                     "type2"=mypanel2,
# 		    "type3"=mypanel3,
# 		    "type4"=mypanel4,
# 		    NULL)
# 
# 
# lower.panel<-switch(lower.panel,
#                     "type2"=mypanel2,
# 		    "type3"=mypanel3,
# 		    "type4"=mypanel4,
# 		    NULL)
# 
# if(plot){
# 
#     if (is.null(clust)) {
#         if (!is.null(obj)) {
#             pairs(S, panel = mypanel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("Contours of Mixture using EmSkew:",toupper(obj$distr), "Distribution") )
#         }
#         else {
#             pairs(S, upper.panel = upper.panel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("The heatmap pairwise plots of the data"))
#         }
#     }
#     else pairs(S, pch = ".", col = clust, row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,
# 	main=paste("Clustering using EmSkew:",toupper(distr), "Distribution") )
# 
# 
# 
# }
# 
# 
# if(path!='') {
# 
# png(paste(path,'/',title,".png",sep=''),width=512,height=512)
# 
#     if (is.null(clust)) {
#         if (!is.null(obj)) {
#             pairs(S, panel = mypanel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("Contours of EmSkew:",toupper(obj$distr), "Distribution") )
#         }
#         else {
#             pairs(S, upper.panel = upper.panel, lower.panel = lower.panel, 
#                 row1attop = attop, diag.panel = diag.panel,
# 		main=paste("The heatmap pairwise plots of the data"))
#         }
#     }
#     else pairs(S, pch = ".", col = clust, row1attop = attop, 
#         lower.panel = lower.panel, diag.panel = diag.panel,
# 	main=paste("Clustering using EmSkew:",toupper(distr), "Distribution") )
# 
# dev.off()
# }
# 
# }
# 
# 
# ddmvn<-function(dat,n,p,mean = rep(0,p),cov = diag(p)                   )
# {
# exp(ddmix(dat,n,p,1,"mvn", mean,cov,0,rep(0,p)))
# 
# }
# 
# ddmvt<-function(dat,n,p,mean = rep(0,p),cov=diag(p),nu=4                )
# {
# exp(ddmix(dat,n,p,1, "mvt", mean,cov,nu,rep(0,p)))
# }
# 
# ddmsn<-function(dat,n,p,mean=rep(0,p),cov=diag(p),       del = rep(0,p))
# {
# exp(ddmix(dat,n,p,1, "msn", mean,cov,0,del))
# }
# 
# ddmst<-function(dat,n,p,mean = rep(0,p),cov=diag(p),nu=4,del = rep(0,p))
# {
# exp(ddmix(dat,n,p,1, "mst", mean,cov,nu,del))
# }
# 
# 
# 
# 
# 
# # rdmvn is a wrapper of the function rmvnorm from R package "mvtnorm"
# 
# 
# rdmvn<-function (n, p,mean = rep(0,p), cov = diag(p)) 
# {
# cov<-as.matrix(cov)
#     
# if (nrow(cov) != ncol(cov)) {
#         stop("cov must be a square matrix")
#     }
#     if (length(mean) != nrow(cov)) {
#         stop("mean and cov have non-conforming size")
#     }
# 
# rmvnorm(n, mean = mean, sigma = cov,method="chol")
# 
# }
# 
# 
# 
# rdmvt<-function(n,p,mean = rep(0,p),cov=diag(p),nu=3)
# {
# cov<-as.matrix(cov)
# u<-rgamma(n,nu/2,nu/2)
# t(t(rdmvn(n,p,cov=cov)/sqrt(u))+mean)
# }
# 
# 
# 
# 
# rdemmix2<-function(n,p,g,distr,pro,mu,sigma,dof=NULL,delta=NULL)
# {
# 
# n0 <- table(sample(1:g, n, replace = TRUE, prob = pro))
# nn <- n0
# if(length(nn) <g) {
# nn <- rep(0,g)
# for(i in as.numeric(names(n0)))
# nn[i] <- n0[paste(i)]
# }
# 
# names(nn) <- NULL
# 
# rdemmix(nn,p,g,distr,mu,sigma,dof,delta)
# 
# }
# 
# rdemmix3<-function(n,p,g,distr,pro,mu,sigma,dof=NULL,delta=NULL)
# {
# 
# if(length(pro) != g)
# stop(paste("pro should be a ",g, " vector!"))
# 
# n0 <- table(sample(1:g, n, replace = TRUE, prob = pro))
# nn <- n0
# if(length(nn) <g) {
# nn <- rep(0,g)
# for(i in as.numeric(names(n0)))
# nn[i] <- n0[paste(i)]
# }
# 
# names(nn) <- NULL
# 
# dat <- rdemmix(nn,p,g,distr,mu,sigma,dof,delta)
# 
# list(data = dat, cluster = rep(1:g,nn) )
# 
# }
# 
# 
# rdemmix<-function(nvect,p,g,distr,mu,sigma,dof=NULL,delta=NULL)
# {
# 
# if(length(c(nvect))!=g) stop("nvect should be a vector")
# 
# ndist<-switch(tolower(distr),"mvn"=1,"mvt"=2,"msn"=3,"mst"=4,5)
# 
# if(ndist>4) 
# stop("the model specified is not available yet")
# 
# if(is.null(dof))
# dof <- rep(4,g)
# 
# if(is.null(delta))
# delta <- array(0,c(p,g))
# 
# if(length(c(mu)) != (p*g))
# stop(paste("mu should be a ",p, 'by', g, "matrix!"))
# 
# if(length(c(sigma)) != (p*p*g) )
# stop(paste("sigma should be a ",p, 'by', p,'by', g, " array!"))
# 
# if(length(c(dof)) != g)
# stop(paste("dof should be a ",g, " vector!"))
# 
# if(length(c(delta)) != (p*g) )
# stop(paste("delta should be a ",p, "by", g, " array!"))
# 
# # to fix the "g=1" bug,
# 
# mu    = array(mu, c(p,g))
# sigma = array(sigma, c(p,p,g))
# delta = array(delta, c(p,g))
# 
# dat<-array(0,c(10,p))
# 
# mvrand<-function(n,p,ndist,mean,cov,nu,del)
# {
# 
# switch(ndist,
# '1' = rdmvn(n,p,mean=mean,cov=cov              ),
# '2' = rdmvt(n,p,mean=mean,cov=cov,nu=nu        ),
# '3' = rdmsn(n,p,mean=mean,cov=cov,      del=del),
# '4' = rdmst(n,p,mean=mean,cov=cov,nu=nu,del=del))
# }
# 
# 
# if(g>=1)
# for(h in 1:g)
# {
# if(nvect[h]>0)
# dat<-rbind(dat,mvrand(nvect[h],p,ndist,mu[,h],sigma[,,h],dof[h],delta[,h]))
# 
# }
# 
# dat[-(1:10),]
# }
# 
# 
# # BOOTSTRAP functions
# 
# bootstrap <- function(x,n,p,g,distr,ncov,popPAR,B=99, replace=TRUE,itmax=1000,epsilon=1e-5)
# {
# x<-as.matrix(x);
# 
# 
# if(missing(popPAR))
# stop("please run the function EmSkew() first")
# 
# counter <- 0
# nnn <- g*(1 + p + p*p + 1 + p) 
# 
# ret <- array(0, c(B,nnn)) 
# 
# dimnames(ret) <- list(1:B, c(
# paste("pi",1:g,sep=''),
# paste("mu",rep(1:p,g),rep(paste(1:g,sep=''),rep(p,g)),sep=''),
# paste("sigma",rep(paste(rep(1:p,rep(p,p)),rep(1:p,p),sep=''),g),rep(paste(',',1:g,sep=''),rep(p*p,g)),sep=''),
# paste("dof",1:g,sep=''),
# paste("delta",rep(1:p,g),rep(paste(1:g,sep=''),rep(p,g)),sep='')))
# 
# 
# 
# for(i in 1:(2*B) )
# {
# 
# if(replace)
# dat <- x[sample(1:n,n,replace=TRUE),]
# else
# dat <- rdemmix3(n,p,g,distr,popPAR$pro,popPAR$mu,popPAR$sigma,popPAR$dof,popPAR$delta)
# 
# 
# obj <- EmSkewfit2(dat,g, popPAR, distr,ncov,itmax,epsilon)
# 
# if(obj$error > 1) next
# 
# counter <- counter +1 
# 
# ret[counter,] <- c(obj$pro,obj$mu,obj$sigma,obj$dof,obj$delta)
# 
# if(counter >= B) break 
# 
# }
# 
# std <- sqrt(apply(ret[1:counter,],MARGIN=2,FUN= "var"))
# 
# names(std) <- dimnames(ret)[[2]]
# std
# }
# 
# bootstrap.noc <- function(x,n,p,g1,g2,distr,ncov,B=99, replace=TRUE,itmax=1000,epsilon=1e-5)
# {
# 
# x<-as.matrix(x);
# 
# if(g1 >= g2)
# stop("g1 should be less than g2")
# 
# if(g1 < 1)
# stop("g1 should be greater than 0")
# 
# counter <- 0
# 
# vlk <- rep(0,g2-g1+1)
# 
# ret <- array(0,c(B,g2-g1))
# 
# dimnames(ret) <- list(1:B,paste(1+g1:(g2-1),"vs",(g1:(g2-1)),sep=' '))
# 
# lk0 <- -Inf
# # start
# 
# clust <- rep(1,n)
# 
# for( g in g1:g2) {
# 
# counter <- 0
# 
# lk1 <- -Inf
# 
# while(counter < 10) {
# 
# if(g>1)
# clust <- kmeans(x,g,nstart=5)$cluster
# 
# emobj <- EmSkewfit1(x,g,clust, distr,ncov,itmax,epsilon)
# 
# if(emobj$error>1) next
# 
# if(emobj$loglik > lk1) 
# lk1 <- emobj$loglik 
# 
# counter = counter +1
# 
# }
# 
# #save the results for g
# 
# dput(emobj,paste("ReturnOf_g_",g,".ret",sep=''))
# 
# #----------------------
# 
# counter <- 0
# 
# lk0 <- lk1
# 
# vlk[g-g1+1] <- lk0
# 
# if(g < g2) {
# 
# 
# for(i in 1:(2*B) )
# {
# 
# if(replace)
# dat <- x[sample(1:n,n,replace=TRUE),]
# else
# dat <- rdemmix2(n,p,g,distr,emobj$pro,emobj$mu,emobj$sigma,emobj$dof,emobj$delta)
# 
# if(is.null(dat)) stop("I can not generate the data!")
# 
# obj <- EmSkewfit2(dat,g, emobj, distr,ncov,itmax,epsilon)
# 
# if(obj$error > 1) next
# 
# ii <- 0
# 
# lk2 <- -Inf
# 
# while(ii<10) {
# 
# clust <- kmeans(dat,g+1,nstart=5)$cluster
# 
# obj2 <- EmSkewfit1(dat,g+1,clust, distr,ncov,itmax,epsilon)
# 
# ii <- ii + 1
# 
# if(obj2$error>1) next
# 
# if(obj2$loglik > lk2)
# lk2 <- obj2$loglik 
# 
# } #end ii loop
# 
# 
# counter <- counter +1 
# 
# ret[counter,g-g1+1] <- -2*(obj$loglik-lk2)
# 
# if(counter >= B) break 
# 
# } #end i loop
# } # end g loop
# 
# }# end if
# 
# pvalue <- rep(0,g2-g1)
# 
# for(i in 1:(g2-g1))
# {
# pvalue[i] <- sum(ret[,i] < 2*(vlk[i+1]-vlk[i]))/B
# }
# 
# list(ret=ret,vlk=vlk,pvalue=pvalue)
# }
# # mahalonobis distance
# 
# mahalonobis<-function(p, g, mu, sigma) 
# {
# 
# 
# obj<-.C('mahalonobis_',PACKAGE="EMMIXskew",
# as.integer(p),as.integer(g),as.double(mu),as.double(sigma), 
# dist = double(g*g), error = integer(1)) 
# 
# if(obj$error) stop("") 
# 
# matrix(obj$dist, ncol=g)
# 
# 
# }
# 
# 
# intradist<-function(dat,g,sigma, clust, tau) 
# {
# dat<-as.matrix(dat)
# 
# intraobj<-.C('intradist_',PACKAGE="EMMIXskew",
# as.double(dat),as.integer((n=nrow(dat))),as.integer((m=ncol(dat))),
# as.integer(g),as.integer(clust),as.double(sigma),as.double(tau),
# dist1=double(g+1),dist2 = double(g+1), error = integer(1)) 
# 
# list(error=intraobj$error,dist1 = intraobj$dist1[1:g],dist2 = intraobj$dist2[1:g],
# OverallIntraDist1=intraobj$dist1[g+1],OverallIntraDist2=intraobj$dist2[g+1])
# }
# 
# 
# interdist<-function(dat,g,sigma, clust, tau) 
# {
# 
# dat<-as.matrix(dat)
# 
# 
# interobj<-.C('interdist_',PACKAGE="EMMIXskew",
# as.double(dat),as.integer((n=nrow(dat))),as.integer((m=ncol(dat))),
# as.integer(g),as.integer(clust),as.double(sigma),as.double(tau),
# dist1=double(g*g+1),dist2 = double(g*g+1), error = integer(1)) 
# 
# 
# 
# list(error=interobj$error,dist1 = matrix(interobj$dist1[1:(g*g)],ncol=g),
# dist2 = matrix(interobj$dist2[1:(g*g)],ncol=g), 
# OverallInterDist1 = interobj$dist1[(g*g)+1],
# OverallInterDist2 = interobj$dist2[(g*g)+1]   )
# 
# }
# 
# 
# #----------------------------------
# # U\V |  V_1  V_2 ... V_C  | sums
# # ---------------------------------
# # U_1 |  n_11 n_12... n_1c | a_1
# # U_2 |  n_21 n_22... n_2c | a_2
# # .
# # .
# # .
# # U_R |  n_r1 n_r2... n_rc | a_r
# #---------------------------------
# #sums  |  b_1  b_2 ... b_c  | N
# #---------------------------------
# 
# rand.index<- function(LabelA,LabelB) {
# 
# 
# u <- unclass(as.ordered(LabelA))
# v <- unclass(as.ordered(LabelB))
# 
# if((N <- length(u)) != length(v))
# stop("Label A and B does not match!")
# 
# #Adjusted Rand Index (ARI)
# 
# row <- max(u)
# col <- max(v)
# 
# nvect <- array(0,c(row,col))
# 
# for(i in 1:row) {
#  for(j in 1:col) {
#    nvect[i,j]<-sum(u==i&v==j)
# 
# }}
# 
# SumsA <- rowSums(nvect)
# SumsB <- colSums(nvect)
# 
# a = 0
# for(i in 1:row)
# a=a+choose(SumsA[i],2)
# 
# b = 0
# for(j in 1:col)
# b=b+choose(SumsB[j],2)
# 
# c <- a*b/choose(N,2)
# 
# d = 0
# for(i in 1:row) {
#  for(j in 1:col) {
#    d=d+choose(nvect[i,j],2)
# }}
# 
# #Adjusted Rand INdex
# arj <- (d-c)/((a+b)/2-c)
# 
# #Rand Index (RI)
# 
# a=d
# 
# b=0
# 
# for(l in 1:row) {
# 
# for(i in 1:(col-1)) {
#  for(j in (i+1):col) {
#    b=b+ nvect[l,i]*nvect[l,j]
# 
# }}
# 
# }
# 
# c=0
# 
# for(l in 1:col) {
# 
# for(i in 1:(row-1)) {
#  for(j in (i+1):row) {
#    c=c+ nvect[i,l]*nvect[j,l]
# 
# }}
# 
# }
# 
# #d= choose(N,2)-a-b-c
# 
# #rad= (a+d)/choose(N,2)
# 
# rad= (choose(N,2)-b-c)/choose(N,2)
# 
# ind <- c(rad,arj)
# 
# names(ind) <- c("Rand Index (RI)","Adjusted Rand Index (ARI)")
# 
# ind
# }
# 
# 
# inverse <-function(sigma,p)
# {
# if(length(c(sigma))!=p*p | ncol(sigma)!=p)
# stop("sigma should be p by p matrix")
# 
# obj <- .Fortran('inverse3',PACKAGE="EMMIXskew",
# as.double(sigma),inv=double(p*p), 
# det=double(1),as.integer(p), error = integer(1),
# count = integer(1),index = integer(p))
# 
# if(obj$error) stop("")
# a <- array(obj$inv,c(p,p))
# as.matrix(t(a)%*%a)
# }
# 
# tau2clust<-function(tao)
# {
# apply(tao,FUN=which.max,MARGIN=1)
# }
# 
# getcov <-function(msigma,sumtau,n,p,g,ncov)
# {
# sigma<-array(0,c(p,p))
# if( (ncov==1)|(ncov==2))
# {
# for(h in 1:g)
# sigma<-sigma+sumtau[h]*msigma[,,h]
# sigma<-as.matrix(sigma/n)
# 
# if(ncov==2)
# sigma<-diag(c(diag(sigma)),p)
# for(h in 1:g)
# msigma[,,h]=sigma
# }
# 
# if(p>1)
# {
# if(ncov==4)
# for(h in 1:g)
# msigma[,,h]<-diag(c(diag(msigma[,,h])),p)
# 
# if(ncov==5)
# for(h in 1:g)
# msigma[,,h]<-diag(sum(diag(msigma[,,h]))/p,p)
# }
# 
# msigma
# }
# 
# 
# mvt.dof <-
# function(sumtau,sumlnv,lx=2+1e-4,ux=200)
# {
# 
# if(sumtau <=2)
# return(4L)
# 
# f<-function(v,sumlnv,sumtau) 
# {
# sumtau*( log(v/2)-digamma(v/2)+1)+ sumlnv
# }
# 
# if(f(lx,sumlnv,sumtau)*f(ux,sumlnv,sumtau)>0)
# return(ux)
# else
# (uniroot(f,c(lx,ux),sumlnv=sumlnv,sumtau=sumtau)$root)
# }
# 
# 
# 
# error.rate<-function(clust1,clust2)
# {
# 
# clust1 <- unclass(as.ordered(clust1))
# clust2 <- unclass(as.ordered(clust2))
# 
# if((n=length(clust1))!=length(clust2))
# {warning("error: length not equal");return}
# 
# if( (g=length(table(clust1)))!=length(table(clust2)))
# {warning("the number of clusters are not equal");return}
# 
# permute<-function(a)
# {
# n<-length(a)
# if(n==1)
# f<-a
# else
# {
# nm<-gamma(n)
# f<-array(0,c(n,n*nm))
# j<-1
# 
# for(i in a)
# {
#  f[1, (j-1)*nm+1:nm]<-i
#  f[-1,(j-1)*nm+1:nm]<-permute(setdiff(a,i))
#  j<-j+1
# }
# }
# 
# f
# }
# 
# 
# #
# id<-1:n
# 
# cmb<-permute(1:g)
# 
# nperm<-ncol(cmb)
# 
# rate<-rep(0,nperm)
# 
# #
# for(i in 1:nperm)
# {
# 
# tmp<-rep(0,g)
# 
# tc<-rep(0,n)
# 
# for(j in 1:g)
# tc[clust2==j]=cmb[j,i]
# 
# for(j in 1:g)
# {  
# tmp1<-0 
# 
# for(k in (1:g)[-j])
#         tmp1<-tmp1+length(intersect(id[clust1==j],id[tc==k]))
# 
# tmp[j]<-tmp1
# }
# 
# rate[i]<-sum(tmp)/n
# }
# 
# min(rate)
# }
# 
# 
# #end


