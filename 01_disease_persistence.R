# -------------------------- required packages --------------------------

required_packages <- c("ggplot2", "scales")

missing_packages <- required_packages[
  !vapply(
    required_packages,
    function(pkg) requireNamespace(pkg, quietly = TRUE),
    logical(1)
  )
]

if (length(missing_packages) > 0L) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org"
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

library(grid)

set.seed(20261016)

# ----------------------------- parameters -----------------------------

Lam <- 8.0
mu <- 0.75
eps <- 0.50

g1 <- c(-0.30, 0.50, 0.20)

Phi <- 1e-3 * matrix(
  c(
    3, 2, 1,
    2, 3, 2,
    1, 2, 4
  ),
  nrow = 3,
  byrow = TRUE
)

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

# Raw drift constants.
# These are used for the Brownian+TS model.
# The Brownian-only model uses the corrected parameters a_i.
mu_raw <- c(
  mu - g1[1],
  mu + eps + 1 - g1[2],
  mu - g1[3]
)

# Calculate a_i = mu_i - integral eta_i(u) nu(du).
# The logarithmic jump increment in direction j is
# zeta[i] * s_dir[j, i] * r.
# Hence eta_i(u) = exp(zeta[i] * s_dir[j, i] * r) - 1.
eta_mean <- vapply(seq_len(3L), function(i) {
  sum(
    w_dir *
      gamma(-alpha) *
      (
        (b_ts - zeta[i] * s_dir[, i])^alpha -
          b_ts^alpha
      )
  )
}, numeric(1))

a_vec <- mu_raw - eta_mean

names(a_vec) <- c("a1", "a2", "a3")

cat("Raw drift parameters mu_i:\n")
print(mu_raw)

cat("Jump corrections integral eta_i(u) nu(du):\n")
print(eta_mean)

cat("Corrected parameters a_i:\n")
print(a_vec)

# Two beta regimes
betas <- c(0.232, 0.316)

T_end <- 80.0
dt <- 0.005
rec_dt <- 0.25
n_path <- 1000L

# Initial value
x0 <- c(
  S = 5.55,
  I = 0.50,
  R = 0.10
)

# ----------------------- tempered-stable sampler ----------------------

# Positive alpha-stable random variable
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

# Exact tempered-stable increment
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

# Three-dimensional logarithmic TS increment
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
  
  # Use corrected drift for Brownian-only simulation.
  # Retain the raw drift for Brownian+TS simulation.
  mu_drift <- if (driver == "Brownian") {
    a_vec
  } else {
    mu_raw
  }
  
  set.seed(seed)
  
  n_step <- as.integer(
    round(T_end / dt)
  )
  
  rec_every <- as.integer(
    round(rec_dt / dt)
  )
  
  n_rec <- n_step %/% rec_every
  
  # Cholesky factor of Phi
  chol_phi <- chol(Phi)
  
  # Log-coordinate state:
  # Y = (log(S), log(I), log(R))
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
    
    # Correlated Brownian increment
    Z <- matrix(
      rnorm(n_path * 3L),
      nrow = n_path,
      ncol = 3L
    )
    
    dW <- sqrt(dt) * (Z %*% chol_phi)
    
    # Brownian-only or Brownian+TS
    if (driver == "Brownian+TS") {
      
      dJ <- ts_log_increment(
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
    
    X <- exp(Y)
    
    S <- X[, 1]
    I <- X[, 2]
    R <- X[, 3]
    
    # Log-coordinate drift
    drift <- cbind(
      Lam / S -
        mu_drift[1] -
        beta * I -
        Phi[1, 1] / 2,
      
      beta * S -
        mu_drift[2] -
        Phi[2, 2] / 2,
      
      I / R -
        mu_drift[3] -
        Phi[3, 3] / 2
    )
    
    # Euler step
    Y <- Y +
      drift * dt +
      dW +
      dJ
    
    # Save the current state
    if (k %% rec_every == 0L) {
      
      ir <- ir + 1L
      
      rec_t[ir] <- k * dt
      
      rec[ir, , ] <- exp(Y)
    }
  }
  
  list(
    time = rec_t,
    values = rec,
    beta = beta,
    driver = driver,
    seed = seed,
    dt = dt,
    n_path = n_path
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
  
  do.call(
    rbind,
    out
  )
}

# ------------------------------ simulation -----------------------------

all_runs <- list()
all_summary <- list()

counter <- 0L

for (b0 in betas) {
  
  for (drv in c("Brownian", "Brownian+TS")) {
    
    counter <- counter + 1L
    
    sim <- simulate_sir(
      beta = b0,
      driver = drv,
      T_end = T_end,
      dt = dt,
      rec_dt = rec_dt,
      n_path = n_path,
      x0 = x0,
      seed = 7000L + counter
    )
    
    all_runs[[counter]] <- sim
    
    all_summary[[counter]] <- summarise_run(sim)
    
    message(
      sprintf(
        "finished %s, beta = %.3f",
        drv,
        b0
      )
    )
  }
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

# Format logarithmic ticks as 10^k
pow10_labels <- function(x) {
  parse(
    text = paste0(
      "10^",
      round(log10(x))
    )
  )
}

make_panel <- function(
    b0,
    variable,
    show_x = FALSE
) {
  
  d <- summary_df[
    summary_df$beta == b0 &
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
      size = 0.65
    ) +
    scale_colour_manual(
      values = cols
    ) +
    scale_fill_manual(
      values = cols
    ) +
    labs(
      x = NULL,
      y = if (b0 == betas[1]) variable else NULL
    ) +
    theme_bw(
      base_size = 9
    ) +
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(
        colour = "grey88",
        size = 0.25
      ),
      plot.margin = margin(
        3,
        4,
        2,
        3
      ),
      plot.title = element_text(
        hjust = 0.5,
        size = 10
      ),
      axis.title = element_text(
        size = 9
      ),
      axis.text = element_text(
        size = 8
      ),
      axis.text.x = if (show_x) {
        element_text(size = 8)
      } else {
        element_blank()
      },
      axis.ticks.x = if (show_x) {
        element_line()
      } else {
        element_blank()
      }
    )
  
  if (variable == "S") {
    
    p <- p +
      scale_y_continuous(
        expand = expansion(
          mult = c(
            0.03,
            0.05
          )
        )
      )
    
  } else {
    
    if (variable == "I") {
      ybreaks <- 10^c(
        -9,
        -6,
        -3,
        0,
        1
      )
    } else {
      ybreaks <- 10^c(
        -6,
        -3,
        0,
        1
      )
    }
    
    p <- p +
      scale_y_log10(
        breaks = ybreaks,
        labels = pow10_labels
      )
    
    if (b0 == 0.232) {
      
      p <- p +
        geom_hline(
          yintercept = 1e-3,
          linetype = "dashed",
          size = 0.45,
          colour = "grey25"
        )
    }
  }
  
  if (variable == "S") {
    
    p <- p +
      ggtitle(
        bquote(
          beta == .(b0)
        )
      )
  }
  
  p +
    coord_cartesian(
      xlim = c(
        0,
        T_end
      )
    )
}

# First row
p11 <- make_panel(
  0.232,
  "S",
  show_x = FALSE
)

p12 <- make_panel(
  0.316,
  "S",
  show_x = FALSE
)

# Second row
p21 <- make_panel(
  0.232,
  "I",
  show_x = FALSE
)

p22 <- make_panel(
  0.316,
  "I",
  show_x = FALSE
)

# Third row
p31 <- make_panel(
  0.232,
  "R",
  show_x = TRUE
)

p32 <- make_panel(
  0.316,
  "R",
  show_x = TRUE
)

# -------------------------- arrange six panels --------------------------

draw_figure <- function(
    file = NULL,
    device = c("png", "pdf")
) {
  
  device <- match.arg(device)
  
  if (!is.null(file)) {
    
    if (device == "png") {
      
      png(
        file,
        width = 2160,
        height = 2040,
        res = 300,
        bg = "white"
      )
    }
    
    if (device == "pdf") {
      
      pdf(
        file,
        width = 7.2,
        height = 6.8,
        useDingbats = FALSE,
        paper = "special"
      )
    }
    
    on.exit(
      dev.off(),
      add = TRUE
    )
  }
  
  grid.newpage()
  
  pushViewport(
    viewport(
      layout = grid.layout(
        nrow = 5,
        ncol = 2,
        heights = unit(
          c(
            1,
            1,
            1,
            0.12,
            0.16
          ),
          c(
            "null",
            "null",
            "null",
            "null",
            "null"
          )
        )
      )
    )
  )
  
  panel_list <- list(
    p11,
    p12,
    p21,
    p22,
    p31,
    p32
  )
  
  positions <- list(
    c(1, 1),
    c(1, 2),
    c(2, 1),
    c(2, 2),
    c(3, 1),
    c(3, 2)
  )
  
  for (i in seq_along(panel_list)) {
    
    print(
      panel_list[[i]],
      vp = viewport(
        layout.pos.row = positions[[i]][1],
        layout.pos.col = positions[[i]][2]
      )
    )
  }
  
  # t below the left and right bottom panels
  pushViewport(
    viewport(
      layout.pos.row = 4,
      layout.pos.col = 1:2
    )
  )
  
  grid.text(
    expression(t),
    x = unit(0.25, "npc"),
    y = unit(0.55, "npc"),
    gp = gpar(fontsize = 10)
  )
  
  grid.text(
    expression(t),
    x = unit(0.75, "npc"),
    y = unit(0.55, "npc"),
    gp = gpar(fontsize = 10)
  )
  
  popViewport()
  
  # Legend
  pushViewport(
    viewport(
      layout.pos.row = 5,
      layout.pos.col = 1:2
    )
  )
  
  grid.lines(
    x = unit(
      c(
        0.34,
        0.38
      ),
      "npc"
    ),
    y = unit(
      0.25,
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
      0.39,
      "npc"
    ),
    y = unit(
      0.25,
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
        0.58,
        0.62
      ),
      "npc"
    ),
    y = unit(
      0.25,
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
      0.63,
      "npc"
    ),
    y = unit(
      0.25,
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

draw_figure(
  "brownian_vs_brownian_ts_SIR.png",
  "png"
)

draw_figure(
  "brownian_vs_brownian_ts_SIR.pdf",
  "pdf"
)
