library(MASS)
library(SymTS)
library(cusum)

## ====== Parameters ======

Lambda  <- 8
mu      <- 5.3
beta    <- 6
epsilon <- 0.5
eta     <- 1

## Drift vector

gamma_1 <- -0.3
gamma_2 <- 0.5
gamma_3 <- 0.2

## ====== Covariance matrix L (3x3) ======

L <- matrix(
  c(3, 2, 1,
    2, 3, 2,
    1, 2, 4),
  nrow = 3,
  ncol = 3,
  byrow = TRUE
)

L_1 <- 0.01 * L

## ====== Initial values ======

S0 <- 1.6
I0 <- 0.4
R0 <- 0.04

## ====== Time settings ======

dt   <- 0.01
Tend <- 600

time <- seq(0, Tend, by = dt)
steps <- length(time)

n_inc <- steps - 1

## ====== Jump parameters ======

w <- 0.5

## ====== Direction vector ======

theta_values <- seq(0, 2 * pi, length.out = 7)[-1]

direction_matrix <- matrix(
  0,
  nrow = 3,
  ncol = 6
)

for (p in 1:6) {
  
  theta <- theta_values[p]
  
  direction_matrix[1, p] <- sqrt(1 - w^2) * cos(theta)
  direction_matrix[2, p] <-sqrt(1 - w^2) * sin(theta)
  direction_matrix[3, p] <-w
 }

print(direction_matrix)

ell <- 1

jump_1 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 1 * dt,
  ell = ell,
  mu = 0
)

jump_2 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 5 * dt,
  ell = ell,
  mu = 0
)

jump_3 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 5 * dt,
  ell = ell,
  mu = 0
)

jump_4 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 5 * dt,
  ell = ell,
  mu = 0
)

jump_5 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 1 * dt,
  ell = ell,
  mu = 0
)

jump_6 <- rCTS(
  n_inc,
  alpha = 0.4,
  c = 1 * dt,
  ell = ell,
  mu = 0
)

jump_matrix <- rbind(
  jump_1,
  jump_2,
  jump_3,
  jump_4,
  jump_5,
  jump_6
)

zeta <- c(0.1, 0.1, 0.1)

jump_matrix <- rbind(jump_1, jump_2, jump_3,
                     jump_4, jump_5, jump_6)

zeta <- c(0.1, 0.1, 0.1)

jump_vector <- matrix(
  0,
  nrow = 3,
  ncol = n_inc
)

for (j in 1:6) {
  
  u_j <- outer(
    direction_matrix[, j],
    jump_matrix[j, ]
  )
  
  eta_u_j <- exp(
    sweep(u_j, 1, zeta, "*")
  ) - 1
  
  jump_vector <- jump_vector + eta_u_j
}

## ====== 1. Deterministic solution ======

S_det <- numeric(steps)
I_det <- numeric(steps)
R_det <- numeric(steps)

S_det[1] <- S0
I_det[1] <- I0
R_det[1] <- R0

for (k in 2:steps) {
  
  dS_drift <- Lambda -
    mu * S_det[k - 1] -
    beta * S_det[k - 1] * I_det[k - 1]
  
  dI_drift <- beta * S_det[k - 1] * I_det[k - 1] -
    (mu + epsilon + eta) * I_det[k - 1]
  
  dR_drift <- eta * I_det[k - 1] -
    mu * R_det[k - 1]
  
  S_det[k] <- S_det[k - 1] +
    dS_drift * dt
  
  I_det[k] <- I_det[k - 1] +
    dI_drift * dt
  
  R_det[k] <- R_det[k - 1] +
    dR_drift * dt
  
  S_det[k] <- max(S_det[k], 0)
  I_det[k] <- max(I_det[k], 0)
  R_det[k] <- max(R_det[k], 0)
}

## ====== Pre-generate Brownian motion increments ======

dW_matrix <- matrix(
  0,
  nrow = n_inc,
  ncol = 3
)

for (k in 1:n_inc) {
  
  dW_matrix[k, ] <- mvrnorm(
    n = 1,
    mu = c(0, 0, 0),
    Sigma = L_1 * dt
  )
}

## ====== 2. White noise solution ======

S_wn <- numeric(steps)
I_wn <- numeric(steps)
R_wn <- numeric(steps)

S_wn[1] <- S0
I_wn[1] <- I0
R_wn[1] <- R0

for (k in 2:steps) {
  
  dS_drift <- Lambda -
    mu * S_wn[k - 1] -
    beta * S_wn[k - 1] * I_wn[k - 1]
  
  dI_drift <- beta * S_wn[k - 1] * I_wn[k - 1] -
    (mu + epsilon + eta) * I_wn[k - 1]
  
  dR_drift <- eta * I_wn[k - 1] -
    mu * R_wn[k - 1]
  
  dW1 <- dW_matrix[k - 1, 1]
  dW2 <- dW_matrix[k - 1, 2]
  dW3 <- dW_matrix[k - 1, 3]
  
  S_wn[k] <- S_wn[k - 1] +
    dS_drift * dt +
    S_wn[k - 1] * dW1
  
  I_wn[k] <- I_wn[k - 1] +
    dI_drift * dt +
    I_wn[k - 1] * dW2
  
  R_wn[k] <- R_wn[k - 1] +
    dR_drift * dt +
    R_wn[k - 1] * dW3
  }

## ====== 3. White noise + jump solution ======

S_wnj <- numeric(steps)
I_wnj <- numeric(steps)
R_wnj <- numeric(steps)

S_wnj[1] <- S0
I_wnj[1] <- I0
R_wnj[1] <- R0

for (k in 2:steps) {
  
  dS_drift <- Lambda -
    (mu - gamma_1) * S_wnj[k - 1] -
    beta * S_wnj[k - 1] * I_wnj[k - 1]
  
  dI_drift <- beta * S_wnj[k - 1] * I_wnj[k - 1] -
    (mu + epsilon + eta - gamma_2) *
    I_wnj[k - 1]
  
  dR_drift <- eta * I_wnj[k - 1] -
    (mu - gamma_3) * R_wnj[k - 1]
  
  dW1 <- dW_matrix[k - 1, 1]
  dW2 <- dW_matrix[k - 1, 2]
  dW3 <- dW_matrix[k - 1, 3]
  
  jump_S <- jump_vector[1, k - 1] *
    S_wnj[k - 1]
  
  jump_I <- jump_vector[2, k - 1] *
    I_wnj[k - 1]
  
  jump_R <- jump_vector[3, k - 1] *
    R_wnj[k - 1]
  
  S_wnj[k] <- S_wnj[k - 1] +
    dS_drift * dt +
    S_wnj[k - 1] * dW1 +
    jump_S
  
  I_wnj[k] <- I_wnj[k - 1] +
    dI_drift * dt +
    I_wnj[k - 1] * dW2 +
    jump_I
  
  R_wnj[k] <- R_wnj[k - 1] +
    dR_drift * dt +
    R_wnj[k - 1] * dW3 +
    jump_R
  }

## ====== Create data frame ======

out_df1 <- data.frame(
  time = time,
  S_det = S_det,
  I_det = I_det,
  R_det = R_det,
  S_wn = S_wn,
  I_wn = I_wn,
  R_wn = R_wn,
  S_wnj = S_wnj,
  I_wnj = I_wnj,
  R_wnj = R_wnj
)

## ====== White noise CUSUM ======

dW2_all <- dW_matrix[, 2]

I_wn_mean <- 0
I_wn_sd <- sqrt(L_1[2, 2] * dt)

C_pos <- numeric(length(dW2_all))
C_pos[1] <- 0

for (t in 2:length(dW2_all)) {
  
  C_pos[t] <- max(
    0,
    C_pos[t - 1] +
      ((dW2_all[t] - I_wn_mean) /
         I_wn_sd - 1)
  )
}

time_index <- time[seq_along(dW2_all)]

H <- 2.38

ymax_I <- max(
  I_wn[seq_along(time_index)],
  na.rm = TRUE
)

ymax_C <- max(
  c(C_pos, H),
  na.rm = TRUE
)

xlim_all <- range(
  time_index,
  na.rm = TRUE
)

par(
  mfrow = c(2, 1),
  mar = c(4, 4, 1.2, 1),
  oma = c(0, 0, 0, 0)
)

plot(
  time_index,
  I_wn[seq_along(time_index)],
  type = "l",
  col = "blue",
  lwd = 1.6,
  xlab = "",
  ylab = "I(t)",
  xlim = xlim_all,
  ylim = c(0, ymax_I)
)

legend(
  "topright",
  legend = "White noise",
  col = "blue",
  lty = 1,
  lwd = 1.6,
  bty = "n",
  cex = 0.8
)

plot(
  time_index,
  C_pos,
  type = "l",
  col = "black",
  lwd = 1.6,
  xlab = "Time",
  ylab = "CUSUM",
  xlim = xlim_all,
  ylim = c(0, ymax_C)
)

abline(
  h = H,
  col = "black",
  lty = 2,
  lwd = 2
)

legend(
  "topright",
  legend = c("CUSUM", "Threshold H"),
  col = c("black", "black"),
  lty = c(1, 2),
  lwd = c(1.6, 2),
  bty = "n",
  cex = 0.8
)

## ====== White noise + tempered stable CUSUM ======

sum_jw <- dW_matrix[, 2] + jump_vector[2, ]

C_pos <- numeric(length(sum_jw))
C_pos[1] <- 0

for (t in 2:length(sum_jw)) {
  
  C_pos[t] <- max(
    0,
    C_pos[t - 1] +
      ((sum_jw[t] - I_wn_mean) /
         I_wn_sd - 1)
  )
}

time_index <- time[seq_along(sum_jw)]

ymax_I <- max(
  I_wnj[seq_along(time_index)],
  na.rm = TRUE
)

ymax_C <- max(
  c(C_pos, H),
  na.rm = TRUE
)

xlim_all <- range(
  time_index,
  na.rm = TRUE
)

par(
  mfrow = c(2, 1),
  mar = c(4, 4, 1.2, 1),
  oma = c(0, 0, 0, 0)
)

plot(
  time_index,
  I_wnj[seq_along(time_index)],
  type = "l",
  col = "red",
  lwd = 1.6,
  xlab = "",
  ylab = " I(t)",
  xlim = xlim_all,
  ylim = c(0, ymax_I)
)

legend(
  "topright",
  legend = "White noise +\nTempered stable",
  col = "red",
  lty = 1,
  lwd = 1.6,
  bty = "n",
  cex = 0.8
)

plot(
  time_index,
  C_pos,
  type = "l",
  col = "black",
  lwd = 2,
  xlab = "Time",
  ylab = "CUSUM",
  xlim = xlim_all,
  ylim = c(0, ymax_C)
)

abline(
  h = H,
  col = "black",
  lty = 2,
  lwd = 2
)

legend(
  "topright",
  legend = c("CUSUM", "Threshold H"),
  col = c("black", "black"),
  lty = c(1, 2),
  lwd = c(2, 2),
  bty = "n",
  cex = 0.8
)