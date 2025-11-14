
#!/usr/bin/env Rscript
# ------------------------------------------------------------------
# Robustness test – EmSkew fitting (internal init)  v7
#   • Warns if any per‑element relative error > 1
# ------------------------------------------------------------------
#path <- "C:/opt/workspace/R/EMMIXskew103emmix2018/EMMIXskew/R"
#files <- list.files(path, pattern = "\\.R$", full.names = TRUE)
#sapply(files, source)

suppressPackageStartupMessages(library(EMMIXskew))
library(mvtnorm)

# -------- redirect output -----------------------------------------
logfile <- paste0("_EmSkew_test_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt")
sink(logfile, split = TRUE)
cat("Log file:", logfile, "\n\n")

set.seed(20250501)

# ---------------- utilities ---------------------------------------

rand_cov <- function(p) {
  A <- matrix(rnorm(p * p), p, p)
  S <- crossprod(A)
  D <- diag(1 / sqrt(diag(S)))
  D %*% S %*% D
}

fmt_vec <- function(v, digits = 3) {
  paste(format(round(v, digits), nsmall = digits), collapse = " ")
}

fmt_mat <- function(M, digits = 3) {
  rows <- apply(round(M, digits), 1,
                function(row) paste(format(row, nsmall = digits), collapse = " "))
  return(rows)
}

rel_vec <- function(est, true) abs(est - true) / pmax(1, abs(true))
rel_mat <- function(est, true) abs(est - true) / pmax(1, abs(true))

# ------------- settings -------------------------------------------
dims   <- c(3,5,7)
n_rep  <- 5
n      <- 30000
g      <- 1

total_sim <- length(dims) * n_rep
sim <- 1

cat("==============================================================\n")
cat("ROBUSTNESS TEST – EmSkew skew‑t (internal init) v7\n")
cat("Total simulations:", total_sim, "\n\n")

for (p in dims) {
  for (rep_ix in seq_len(n_rep)) {
    
    cat("--------------------------------------------------------------\n")
    cat(sprintf("[Simulation %d / %d] Dimension p = %d (rep %d)\n",
                sim, total_sim, p, rep_ix))
    
    # ---- true parameters ----------------------------------------
    mu_true    <- runif(p, -3, 3)
    Sigma_true <- rand_cov(p)
    delta_true <- runif(p, -3, 3)
    dof_true   <- sample(3:10, 1)
    
    # ---- simulate data ------------------------------------------
    cat("  + Simulating data ... ")
    tm_sim <- system.time({
      dat <- rdemmix(
        n     = n,
        p     = p,
        g     = g,
        distr = "mst",
        mu    = matrix(mu_true, p, g),
        sigma = array(Sigma_true, dim = c(p, p, g)),
        dof   = dof_true,
        delta = matrix(delta_true, p, g)
      )
    })
    cat(sprintf("done (%.1f s)\n", tm_sim[3]))
    
    # ---- internal initialisation (for display) ------------------
    init_obj <- try(init.mix(dat, g, "mst", ncov = 3,
                             nkmeans = 0, nrandom = 10, nhclust = FALSE, maxloop = 20),
                    silent = TRUE)
    
    # ---- fit model (EmSkew picks its own init) -------------------
    cat("  + Fitting model with EmSkew (internal init) ...\n")
    tm_fit <- system.time({
      fit <- tryCatch(
        EmSkew(
          dat     = dat,
          g       = g,
          distr   = "mst",
          ncov    = 3,
          itmax   = 1000,
          epsilon = 1e-6,
          debug   = TRUE,
          nkmeans = 0,
          nrandom = 10,
          nhclust = FALSE
        ),
        error = function(e) e
      )
    })
    
    if (inherits(fit, "error")) {
      cat(sprintf("    ❌  Fit failed after %.1f s – %s\n\n\n\n\n",
                  tm_fit[3], fit$message))
      sim <- sim + 1L
      next
    } else {
      cat(sprintf("    ✓  Fit completed in %.1f s (logLik = %.2f)\n",
                  tm_fit[3], fit$loglik))
    }
    
    # ---- error calculations -------------------------------------
    err_mu_vec    <- rel_vec(drop(fit$mu), mu_true)
    err_delta_vec <- rel_vec(drop(fit$delta), delta_true)
    err_Sigma_mat <- rel_mat(fit$sigma[,,1], Sigma_true)
    err_dof       <- abs(fit$dof - dof_true) / dof_true
    
    # ---- warning if any error > 1 -------------------------------
    if (any(err_mu_vec > 1)) {
      for (i in which(err_mu_vec > 1)) {
        cat(sprintf("ATTENZIONE: ERRORE RELATIVO > 1 per mu[%d] nella simulazione %d\n",
                    i, sim))
      }
    }
    if (any(err_delta_vec > 1)) {
      for (i in which(err_delta_vec > 1)) {
        cat(sprintf("ATTENZIONE: ERRORE RELATIVO > 1 per delta[%d] nella simulazione %d\n",
                    i, sim))
      }
    }
    if (any(err_Sigma_mat > 1)) {
      idx <- which(err_Sigma_mat > 1, arr.ind = TRUE)
      apply(idx, 1, function(rc) {
        cat(sprintf("ATTENZIONE: ERRORE RELATIVO > 1 per Sigma[%d,%d] nella simulazione %d\n",
                    rc[1], rc[2], sim))
      })
    }
    if (err_dof > 1) {
      cat(sprintf("ATTENZIONE: ERRORE RELATIVO > 1 per dof nella simulazione %d\n",
                  sim))
    }
    
    # ---- report --------------------------------------------------
    if (!inherits(init_obj, "try-error") && !is.null(init_obj)) {
      cat("\n  * INITIAL parameters (generated by EmSkew):\n")
      cat("    mu0    =", fmt_vec(drop(init_obj$mu)), "\n")
      cat("    delta0 =", fmt_vec(drop(init_obj$delta)), "\n")
      cat("    dof0   =", init_obj$dof, "\n")
      cat("    Sigma0 =\n")
      cat(paste(fmt_mat(init_obj$sigma[,,1]), collapse = "\n      "), "\n")
    } else {
      cat("\n  * INITIAL parameters: <not available – init.mix failed>\n")
    }
    
    cat("\n  * TRUE parameters:\n")
    cat("    mu     =", fmt_vec(mu_true), "\n")
    cat("    delta  =", fmt_vec(delta_true), "\n")
    cat("    dof    =", dof_true, "\n")
    cat("    Sigma  =\n")
    cat(paste(fmt_mat(Sigma_true), collapse = "\n      "), "\n")
    
    cat("\n  * FITTED parameters:\n")
    cat("    mu     =", fmt_vec(drop(fit$mu)), "\n")
    cat("    delta  =", fmt_vec(drop(fit$delta)), "\n")
    cat("    dof    =", fit$dof, "\n")
    cat("    Sigma  =\n")
    cat(paste(fmt_mat(fit$sigma[,,1]), collapse = "\n      "), "\n")
    
    cat("\n  * RELATIVE ERRORS (per element):\n")
    cat("    mu     =", fmt_vec(err_mu_vec, 4), "\n")
    cat("    delta  =", fmt_vec(err_delta_vec, 4), "\n")
    cat("    dof    =", sprintf("%.4f", err_dof), "\n")
    cat("    Sigma  =\n")
    cat(paste(fmt_mat(err_Sigma_mat, 4), collapse = "\n      "), "\n")
    
    cat("--------------------------------------------------------------\n")
    cat("\n\n\n\n\n")   # spacing
    
    sim <- sim + 1L
  }
}

cat("All simulations completed.\n")
sink()
