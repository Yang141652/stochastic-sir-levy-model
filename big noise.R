
#!/usr/bin/env Rscript

# ============================================================
# Brownian versus Brownian + Tempered-stable SIR simulation
#
# ONLY beta = 0.232 is considered.
# Both models use the same effective mean rates a_i.
#
# Brownian-only:
#   drift correction = a_i
#
# Brownian + TS:
#   drift correction = a_i + P_i
#   compensated TS increments are used.
#
# sigma22 = 0.3
# The figure contains two panels: I(t) and R(t).
# PDF is saved using cairo_pdf() with embedded fonts.
# ============================================================

required_packages <- c("ggplot2")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org"
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
})

library(grid)

# ----------------------------- parameters -----------------------------

Lam <- 8.0
mu <- 0.75
eps <- 0.50
g1 <- c(-0.30, 0.50, 0.20)

# sigma22 = Phi[2,2] = 0.3
Phi1 <- matrix(
  c(
    3e-3, 2e-3, 1e-3,
    2e-3, 3e-3,  2e-3,
    1e-3, 2e-3, 4e-3
  ),
  nrow = 3,
  byrow = TRUE
)
Phi<-200*Phi1
# Tempered-stable parameters
alpha <- 0.40
b_ts <- 1.00
zeta <- c(0.01, 0.30, 0.01)
omega <- 0.50
c_dir <- c(1, 5, 5, 5, 1, 1)

theta <- 2 * pi * (1:6) / 6

s_dir <- cbind(
  sqrt(1 - omega^2) * cos(theta),
  sqrt(1 - omega^2) * sin(theta),
  rep(omega, 6)
)

w_dir <- c_dir * b_ts^(-alpha)

# Original drift constants
mu_raw <- c(
  mu - g1[1],
  mu + eps + 1 - g1[2],
  mu - g1[3]
)

# -------------------- effective mean rates a_i ------------------------

x_jump <- sweep(
  s_dir,
  MARGIN = 2,
  STATS = zeta,
  FUN = "*"
)

if (any(b_ts - x_jump <= 0)) {
  stop("Invalid tempering parameter.")
}

# J_eta[i] = integral eta_i(u) nu(du)
J_eta <- gamma(-alpha) * colSums(
  sweep(
    (b_ts - x_jump)^alpha - b_ts^alpha,
    MARGIN = 1,
    STATS = w_dir,
    FUN = "*"
  )
)

# m_log[i] = integral ln(1 + eta_i(u)) nu(du)
m_log <- gamma(1 - alpha) *
  b_ts^(alpha - 1) *
  colSums(
    sweep(
      x_jump,
      MARGIN = 1,
      STATS = w_dir,
      FUN = "*"
    )
  )

# P_i = integral [eta_i(u) - ln(1+eta_i(u))] nu(du)
P_jump <- J_eta - m_log

# a_i = mu_i - integral eta_i(u) nu(du)
a_mean <- mu_raw - J_eta

if (any(!is.finite(a_mean)) ||
    any(!is.finite(P_jump)) ||
    any(a_mean <= 0)) {
  stop("Invalid effective mean rates or TS corrections.")
}

cat("\nEffective mean rates and TS corrections:\n")

print(
  data.frame(
    Parameter = c("a1", "a2", "a3"),
    Original_mu = mu_raw,
    J_eta = J_eta,
    m_log = m_log,
    P_i = P_jump,
    a_i = a_mean
  ),
  digits = 10,
  row.names = FALSE
)

# Only beta = 0.232
beta_value <- 0.232

Re0_W <- beta_value * Lam /
  (a_mean[1] * a_mean[2]) -
  Phi[2, 2] / (2 * a_mean[2])

cat(sprintf(
  "\nBrownian-only Re0_W = %.8f\n",
  Re0_W
))

# ----------------------- simulation parameters -----------------------

T_end <- 80.0
dt <- 0.005
rec_dt <- 0.25
n_path <- 1000L

x0 <- c(
  S = 5.55,
  I = 0.50,
  R = 0.10
)

# ----------------------- tempered-stable sampler ----------------------

positive_stable <- function(alpha, n) {
  
  U <- runif(n, 0, pi)
  E <- rexp(n, rate = 1)
  
  (
    sin(alpha * U) / sin(U)^(1 / alpha)
  ) *
    (
      sin((1 - alpha) * U) / E
    )^((1 - alpha) / alpha)
}


ts_increment <- function(alpha, w, b, dt, n) {
  
  scale <- (
    dt * w * abs(gamma(-alpha))
  )^(1 / alpha)
  
  out <- numeric(n)
  todo <- seq_len(n)
  
  while (length(todo) > 0L) {
    
    X <- scale * positive_stable(
      alpha = alpha,
      n = length(todo)
    )
    
    accept <- runif(length(todo)) < exp(-b * X)
    
    if (any(accept)) {
      out[todo[accept]] <- X[accept]
    }
    
    todo <- todo[!accept]
  }
  
  out
}


# Raw logarithmic TS increments
ts_log_increment <- function(dt, n) {
  
  dL <- matrix(
    0,
    nrow = n,
    ncol = 3
  )
  
  for (j in seq_len(nrow(s_dir))) {
    
    rj <- ts_increment(
      alpha = alpha,
      w = w_dir[j],
      b = b_ts,
      dt = dt,
      n = n
    )
    
    dL <- dL + rj %o% s_dir[j, ]
  }
  
  sweep(
    dL,
    MARGIN = 2,
    STATS = zeta,
    FUN = "*"
  )
}


# Compensated logarithmic TS increments
# dL_tilde = dL - m_log * dt
ts_compensated_increment <- function(dt, n) {
  
  dL <- ts_log_increment(
    dt = dt,
    n = n
  )
  
  sweep(
    dL,
    MARGIN = 2,
    STATS = m_log * dt,
    FUN = "-"
  )
}


# ------------------------------ simulator ------------------------------

simulate_sir <- function(
    beta,
    driver = c("Brownian", "Brownian+TS"),
    T_end,
    dt,
    rec_dt,
    n_path,
    x0,
    seed
) {
  
  driver <- match.arg(driver)
  
  set.seed(seed)
  
  # Both models use the same effective mean rates a_i.
  #
  # Brownian:
  #   drift correction = a_i
  #
  # Brownian + TS:
  #   drift correction = a_i + P_i
  #   compensated TS increments are used.
  
  if (driver == "Brownian") {
    
    mu_use <- a_mean
    
  } else {
    
    mu_use <- a_mean + P_jump
    
  }
  
  n_step <- as.integer(
    round(T_end / dt)
  )
  
  rec_every <- as.integer(
    round(rec_dt / dt)
  )
  
  if (abs(n_step * dt - T_end) > 1e-10 ||
      abs(rec_every * dt - rec_dt) > 1e-10 ||
      n_step %% rec_every != 0L) {
    stop("T_end and rec_dt must be compatible with dt.")
  }
  
  n_rec <- n_step %/% rec_every
  
  chol_phi <- chol(Phi)
  
  Y <- matrix(
    rep(log(x0), each = n_path),
    nrow = n_path,
    ncol = 3
  )
  
  rec_t <- numeric(n_rec + 1L)
  
  rec <- array(
    NA_real_,
    dim = c(n_rec + 1L, n_path, 3L),
    dimnames = list(
      NULL,
      NULL,
      c("S", "I", "R")
    )
  )
  
  rec[1, , ] <- exp(Y)
  rec_t[1] <- 0
  
  ir <- 1L
  
  for (k in seq_len(n_step)) {
    
    # Brownian increments
    Z <- matrix(
      rnorm(n_path * 3L),
      nrow = n_path,
      ncol = 3L
    )
    
    dW <- sqrt(dt) * (Z %*% chol_phi)
    
    # Tempered-stable increments
    if (driver == "Brownian+TS") {
      
      dJ <- ts_compensated_increment(
        dt = dt,
        n = n_path
      )
      
    } else {
      
      dJ <- matrix(
        0,
        nrow = n_path,
        ncol = 3L
      )
      
    }
    
    # Current population sizes
    X <- exp(Y)
    
    S <- X[, 1]
    I <- X[, 2]
    R <- X[, 3]
    
    # Logarithmic drift terms
    drift <- cbind(
      
      Lam / S -
        mu_use[1] -
        beta * I -
        Phi[1, 1] / 2,
      
      beta * S -
        mu_use[2] -
        Phi[2, 2] / 2,
      
      I / R -
        mu_use[3] -
        Phi[3, 3] / 2
      
    )
    
    # Update logarithmic SIR variables
    Y <- Y +
      drift * dt +
      dW +
      dJ
    
    # Record simulation results
    if (k %% rec_every == 0L) {
      
      ir <- ir + 1L
      
      rec_t[ir] <- k * dt
      
      rec[ir, , ] <- exp(Y)
      
    }
    
  }
  
  return(
    list(
      time = rec_t,
      values = rec,
      beta = beta,
      driver = driver,
      seed = seed,
      dt = dt,
      n_path = n_path
    )
  )
  
}


# ----------------------------- summaries -------------------------------

summarise_run <- function(sim) {
  
  vals <- sim$values
  
  out <- vector(
    "list",
    length(sim$time) * 3L
  )
  
  z <- 1L
  
  for (j in seq_len(3L)) {
    
    for (i in seq_along(sim$time)) {
      
      q <- quantile(
        vals[i, , j],
        probs = c(
          0.05,
          0.25,
          0.50,
          0.75,
          0.95
        ),
        names = FALSE,
        type = 7
      )
      
      out[[z]] <- data.frame(
        time = sim$time[i],
        variable = c("S", "I", "R")[j],
        q05 = q[1],
        q25 = q[2],
        median = q[3],
        q75 = q[4],
        q95 = q[5],
        beta = sim$beta,
        driver = sim$driver
      )
      
      z <- z + 1L
    }
  }
  
  do.call(rbind, out)
}


# ------------------------- run beta = 0.232 only -----------------------

all_runs <- list()
all_summary <- list()

drivers <- c(
  "Brownian",
  "Brownian+TS"
)

for (j in seq_along(drivers)) {
  
  drv <- drivers[j]
  
  sim <- simulate_sir(
    beta = beta_value,
    driver = drv,
    T_end = T_end,
    dt = dt,
    rec_dt = rec_dt,
    n_path = n_path,
    x0 = x0,
    seed = 7000L + j
  )
  
  all_runs[[drv]] <- sim
  
  all_summary[[drv]] <- summarise_run(sim)
  
  message(
    sprintf(
      "Finished %s, beta = %.3f",
      drv,
      beta_value
    )
  )
}

summary_df <- do.call(
  rbind,
  all_summary
)


# ------------------------------- plotting -------------------------------

cols <- c(
  "Brownian" = "#0072B2",
  "Brownian+TS" = "#D55E00"
)


pow10_labels <- function(x) {
  
  parse(
    text = paste0(
      "10^",
      round(log10(x))
    )
  )
  
}


make_ir_panel <- function(
    beta_value,
    variable
) {
  
  d <- summary_df[
    summary_df$beta == beta_value &
      summary_df$variable == variable,
  ]
  
  p <- ggplot(
    d,
    aes(
      x = time,
      colour = driver,
      fill = driver
    )
  ) +
    
    geom_ribbon(
      aes(
        ymin = q05,
        ymax = q95
      ),
      alpha = 0.16,
      colour = NA
    ) +
    
    geom_ribbon(
      aes(
        ymin = q25,
        ymax = q75
      ),
      alpha = 0.34,
      colour = NA
    ) +
    
    geom_line(
      aes(y = median),
      linewidth = 0.65
    ) +
    
    scale_colour_manual(
      values = cols
    ) +
    
    scale_fill_manual(
      values = cols
    ) +
    
    labs(
      x = expression(t),
      y = variable
    ) +
    
    theme_bw(
      base_size = 9
    ) +
    
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        colour = "grey88",
        linewidth = 0.25
      ),
      plot.margin = margin(
        4,
        4,
        4,
        4
      ),
      axis.title = element_text(
        size = 10
      ),
      axis.text = element_text(
        size = 8
      )
    ) +
    
    scale_y_log10(
      breaks = if (variable == "I") {
        10^c(-9, -6, -3, 0, 1)
      } else {
        10^c(-6, -3, 0, 1)
      },
      labels = pow10_labels
    ) +
    
    coord_cartesian(
      xlim = c(0, T_end)
    )
  
  if (beta_value == 0.232) {
    
    p <- p +
      geom_hline(
        yintercept = 1e-3,
        linetype = "dashed",
        linewidth = 0.45,
        colour = "grey25"
      )
    
  }
  
  p
  
}


draw_ir_figure <- function(beta_value) {
  
  pI <- make_ir_panel(
    beta_value = beta_value,
    variable = "I"
  )
  
  pR <- make_ir_panel(
    beta_value = beta_value,
    variable = "R"
  )
  
  grid.newpage()
  
  pushViewport(
    viewport(
      layout = grid.layout(
        3,
        2,
        heights = unit(
          c(
            0.14,
            1,
            0.16
          ),
          c(
            "null",
            "null",
            "null"
          )
        )
      )
    )
  )
  
  # Top title
  pushViewport(
    viewport(
      layout.pos.row = 1,
      layout.pos.col = 1:2
    )
  )
  
  grid.text(
    bquote(
     sigma[22] == 0.6
    ),
    x = unit(
      0.5,
      "npc"
    ),
    y = unit(
      0.5,
      "npc"
    ),
    gp = gpar(
      fontsize = 11
    )
  )
  
  popViewport()
  
  # Left panel: I(t)
  print(
    pI,
    vp = viewport(
      layout.pos.row = 2,
      layout.pos.col = 1
    )
  )
  
  # Right panel: R(t)
  print(
    pR,
    vp = viewport(
      layout.pos.row = 2,
      layout.pos.col = 2
    )
  )
  
  # Bottom legend
  pushViewport(
    viewport(
      layout.pos.row = 3,
      layout.pos.col = 1:2
    )
  )
  
  grid.lines(
    x = unit(
      c(
        0.30,
        0.34
      ),
      "npc"
    ),
    y = unit(
      0.5,
      "npc"
    ),
    gp = gpar(
      col = cols["Brownian"],
      lwd = 2.2
    )
  )
  
  grid.text(
    "Brownian",
    x = unit(
      0.35,
      "npc"
    ),
    y = unit(
      0.5,
      "npc"
    ),
    just = "left",
    gp = gpar(
      fontsize = 9
    )
  )
  
  grid.lines(
    x = unit(
      c(
        0.53,
        0.57
      ),
      "npc"
    ),
    y = unit(
      0.5,
      "npc"
    ),
    gp = gpar(
      col = cols["Brownian+TS"],
      lwd = 2.2
    )
  )
  
  grid.text(
    "Brownian+Tempered stable",
    x = unit(
      0.58,
      "npc"
    ),
    y = unit(
      0.5,
      "npc"
    ),
    just = "left",
    gp = gpar(
      fontsize = 9
    )
  )
  
  popViewport()
  popViewport()
  
}


# Display only beta = 0.232 in RStudio
if (interactive()) {
  draw_ir_figure(beta_value)
}


# ----------------------- save embedded-font PDF -----------------------

if (!capabilities("cairo")) {
  stop(
    "Cairo PDF support is not available in this R installation."
  )
}

pdf_file <- file.path(
  getwd(),
  "6.pdf"
)

grDevices::cairo_pdf(
  filename = pdf_file,
  width = 7,
  height = 4.5,
  family = "sans",
  onefile = FALSE
)

tryCatch(
  draw_ir_figure(beta_value),
  finally = grDevices::dev.off()
)

cat(
  "\nPDF saved with cairo_pdf(): ",
  pdf_file,
  "\n",
  sep = ""
)


# ---------------- beta = 0.232 first-passage time ----------------

first_passage_summary <- function(
    sim,
    level = 1e-3
) {
  
  # sim$values dimensions:
  # time x path x variable
  # The second variable is I.
  
  I_paths <- sim$values[, , 2]
  
  hit_time <- apply(
    I_paths,
    2,
    function(x) {
      
      id <- which(x <= level)
      
      if (length(id) == 0L) {
        
        return(Inf)
        
      } else {
        
        return(sim$time[id[1]])
        
      }
      
    }
  )
  
  reached <- is.finite(hit_time)
  hit_finite <- hit_time[reached]
  
  if (length(hit_finite) == 0L) {
    
    return(
      data.frame(
        beta = sim$beta,
        driver = sim$driver,
        level = level,
        n_total = length(hit_time),
        n_reached = 0,
        n_not_reached = length(hit_time),
        reached_fraction = 0,
        earliest = NA_real_,
        latest = NA_real_,
        interval_95_lower = NA_real_,
        interval_95_upper = NA_real_,
        median_reached = NA_real_,
        median_all_with_censoring = Inf
      )
    )
    
  }
  
  q <- quantile(
    hit_finite,
    probs = c(
      0.05,
      0.50,
      0.95
    ),
    names = FALSE
  )
  
  data.frame(
    beta = sim$beta,
    driver = sim$driver,
    level = level,
    n_total = length(hit_time),
    n_reached = sum(reached),
    n_not_reached = sum(!reached),
    reached_fraction = mean(reached),
    earliest = min(hit_finite),
    latest = max(hit_finite),
    interval_95_lower = q[1],
    interval_95_upper = q[3],
    median_reached = q[2],
    median_all_with_censoring = median(hit_time)
  )
  
}


fpt_results <- do.call(
  rbind,
  lapply(
    all_runs,
    first_passage_summary,
    level = 1e-3
  )
)

cat("\nFirst-passage time summary: beta = 0.232\n")
print(fpt_results)

fpt_file <- file.path(
  getwd(),
  "beta_0.232_I_first_passage_summary.csv"
)

write.csv(
  fpt_results,
  fpt_file,
  row.names = FALSE
)

cat(
  "\nFirst-passage summary saved: ",
  fpt_file,
  "\n",
  sep = ""
)

cat("\nSimulation completed for beta = 0.232 only.\n")
