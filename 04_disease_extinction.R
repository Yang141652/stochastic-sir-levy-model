library(MASS)
library(SymTS)
library(cusum)

## ====== Parameters ======

Lambda  <- 8       # Recruitment rate
mu      <- 5.3     # Natural mortality rate
beta    <- 4.8     # Transmission coefficient
epsilon <- 0.5     # Disease-induced mortality rate
eta     <- 1       # Dimensionless recovery rate

## Drift vector
gamma_1 <- -0.3
gamma_2 <- 0.5
gamma_3 <- 0.2

## ====== Covariance matrix L (3x3) ======

L <- matrix(c(3, 2, 1,
              2, 3, 2,
              1, 2, 4),
            nrow = 3, ncol = 3, byrow = TRUE)

L_1 <- 0.01 * L

## ====== Initial values ======

S0 <- 1.6
I0 <- 0.4
R0 <- 0.04

## ====== Time settings ======

dt   <- 0.01       # Step size
Tend <- 600       # Total time
time <- seq(0, Tend, by = dt)
steps <- length(time)

## Number of random increments required

n_inc <- steps - 1

## ====== Jump parameters ======

w <- 0.5          # w ∈ [0,1]

## ====== Direction vector ======

theta_values <- seq(0, 2*pi, length.out = 7)[-1]

# Create the direction matrix with 3 rows and 6 columns

direction_matrix <- matrix(0, nrow = 3, ncol = 6)

for (p in 1:6) {
  theta <- theta_values[p]
  direction_matrix[1, p] <-  sqrt(1 - w^2) * cos(theta)
  direction_matrix[2, p] <- sqrt(1 - w^2) * sin(theta)
  direction_matrix[3, p] <- w
}

# Print the direction matrix

print(direction_matrix)

ell <- 1

jump_1 <- rCTS(n_inc, alpha = 0.4, c = 1 * dt, ell = ell, mu = 0)
jump_2 <- rCTS(n_inc, alpha = 0.4, c = 5 * dt, ell = ell, mu = 0)
jump_3 <- rCTS(n_inc, alpha = 0.4, c = 5 * dt, ell = ell, mu = 0)
jump_4 <- rCTS(n_inc, alpha = 0.4, c = 5 * dt, ell = ell, mu = 0)
jump_5 <- rCTS(n_inc, alpha = 0.4, c = 1 * dt, ell = ell, mu = 0)
jump_6 <- rCTS(n_inc, alpha = 0.4, c = 1 * dt, ell = ell, mu = 0)

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
  
  S_det[k] <- S_det[k - 1] + dS_drift * dt
  I_det[k] <- I_det[k - 1] + dI_drift * dt
  R_det[k] <- R_det[k - 1] + dR_drift * dt
  
  S_det[k] <- max(S_det[k], 0)
  I_det[k] <- max(I_det[k], 0)
  R_det[k] <- max(R_det[k], 0)
}

## ====== Pre-generate Brownian motion increments ======

dW_matrix <- matrix(0, nrow = n_inc, ncol = 3)

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
  
  # Drift terms
  
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
  
  
  S_wn[k] <- max(S_wn[k], 0)
  I_wn[k] <- max(I_wn[k], 0)
  R_wn[k] <- max(R_wn[k], 0)
}

## ====== 3. White noise + jump solution ======

S_wnj <- numeric(steps)
I_wnj <- numeric(steps)
R_wnj <- numeric(steps)

S_wnj[1] <- S0
I_wnj[1] <- I0
R_wnj[1] <- R0

for (k in 2:steps) {
  
  # Drift terms
  
  dS_drift <- Lambda -
    (mu - gamma_1) * S_wnj[k - 1] -
    beta * S_wnj[k - 1] * I_wnj[k - 1]
  
  dI_drift <- beta * S_wnj[k - 1] * I_wnj[k - 1] -
    (mu + epsilon + eta - gamma_2) *
    I_wnj[k - 1]
  
  dR_drift <- eta * I_wnj[k - 1] -
    (mu - gamma_3) * R_wnj[k - 1]
  
  # Use the same Brownian motion increments as in the white-noise model
  
  dW1 <- dW_matrix[k - 1, 1]
  dW2 <- dW_matrix[k - 1, 2]
  dW3 <- dW_matrix[k - 1, 3]
  
  # Obtain jump terms
  
  jump_S <- jump_vector[1, k - 1] *
    S_wnj[k - 1]
  
  jump_I <- jump_vector[2, k - 1] *
    I_wnj[k - 1]
  
  jump_R <- jump_vector[3, k - 1] *
    R_wnj[k - 1]
  
  # Euler-Maruyama update with white noise and jumps
  
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
  
  S_wnj[k] <- max(S_wnj[k], 0)
  I_wnj[k] <- max(I_wnj[k], 0)
  R_wnj[k] <- max(R_wnj[k], 0)
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

par(mfrow = c(1, 1),
    mar = c(4, 4, 2, 1),
    oma = c(0, 0, 2, 0))

# Comparison of S(t)

plot(out_df1$time, out_df1$S_det,
     type = "l", col = "black", lwd = 2,
     xlab = "Time ",
     ylab = "S(t)",
     ylim = range(c(0,
                    out_df1$S_det,
                    out_df1$S_wn,
                    out_df1$S_wnj)))

lines(out_df1$time, out_df1$S_wn,
      col = "blue", lwd = 1.5)

lines(out_df1$time, out_df1$S_wnj,
      col = "red", lwd = 1.5)

legend("topright",
       legend = c("Deterministic",
                  "White noise",
                  "White noise + \nTempered stable"),
       col = c("black", "blue", "red"),
       lwd = c(2, 1.5, 1.5),
       bty = "n",
       cex = 0.8)

# Comparison of I(t)

plot(out_df1$time, out_df1$I_det,
     type = "l", col = "black", lwd = 2,
     xlab = "Time ",
     ylab = " I(t)",
     ylim = range(c(out_df1$I_det,
                    out_df1$I_wn,
                    out_df1$I_wnj)))

lines(out_df1$time, out_df1$I_wn,
      col = "blue", lwd = 1.5)

lines(out_df1$time, out_df1$I_wnj,
      col = "red", lwd = 1.5)

legend("topright",
       legend = c("Deterministic",
                  "White noise",
                  "White noise + \nTempered stable"),
       col = c("black", "blue", "red"),
       lwd = c(2, 1.5, 1.5),
       bty = "n",
       cex = 0.8)

# Comparison of R(t)

plot(out_df1$time, out_df1$R_det,
     type = "l", col = "black", lwd = 2,
     xlab = "Time",
     ylab = "R(t)",
     ylim = range(c(out_df1$R_det,
                    out_df1$R_wn,
                    out_df1$R_wnj)))

lines(out_df1$time, out_df1$R_wn,
      col = "blue", lwd = 1.5)

lines(out_df1$time, out_df1$R_wnj,
      col = "red", lwd = 1.5)

legend("topright",
       legend = c("Deterministic",
                  "White noise",
                  "White noise + \nTempered stable"),
       col = c("black", "blue", "red"),
       lwd = c(2, 1.5, 1.5),
       bty = "n",
       cex = 0.8)



## ====== Time averages ======

avg_S_wnj <- numeric(steps)
avg_I_wnj <- numeric(steps)
avg_R_wnj <- numeric(steps)

avg_S_wnj[1] <- NA
avg_I_wnj[1] <- NA
avg_R_wnj[1] <- NA

for (k in 2:steps) {
  
  current_time <- time[k]
  
  integral_S_wnj <- sum(
    (S_wnj[1:(k - 1)] + S_wnj[2:k]) * dt / 2
  )
  
  integral_I_wnj <- sum(
    (I_wnj[1:(k - 1)] + I_wnj[2:k]) * dt / 2
  )
  
  integral_R_wnj <- sum(
    (R_wnj[1:(k - 1)] + R_wnj[2:k]) * dt / 2
  )
  
  avg_S_wnj[k] <- integral_S_wnj / current_time
  avg_I_wnj[k] <- integral_I_wnj / current_time
  avg_R_wnj[k] <- integral_R_wnj / current_time
}

## ====== Create data frame ======

out_df2 <- data.frame(
  time = time,
  S_wnj = S_wnj,
  I_wnj = I_wnj,
  R_wnj = R_wnj,
  avg_S_wnj = avg_S_wnj,
  avg_I_wnj = avg_I_wnj,
  avg_R_wnj = avg_R_wnj
)

par(
  mfrow = c(3, 1),
  mar = c(2, 4, 1, 1),
  oma = c(4, 5, 1, 1)
)

## ---------- 1) S(t) ----------

yS_lim <- range(
  out_df2$avg_S_wnj,
  na.rm = TRUE
)

plot(
  out_df2$time,
  out_df2$avg_S_wnj,
  type = "l",
  lwd = 2,
  col = "blue",
  xlab = "",
  ylab = "",
  ylim = yS_lim,
  yaxt = "n"
)

axis(2, at = pretty(yS_lim))

legend(
  "bottomright",
  inset = c(0.01, 0.15),
  bty = "n",
  lwd = 2,
  legend = expression(
    paste(frac(1, t) * integral(S(r) * dr, 0, t))
  ),
  col = "blue"
)

## ---------- 2) I(t) ----------

yI_lim <- range(
  out_df2$avg_I_wnj,
  na.rm = TRUE
)

plot(
  out_df2$time,
  out_df2$avg_I_wnj,
  type = "l",
  lwd = 2,
  col = "blue",
  xlab = "",
  ylab = "",
  ylim = yI_lim,
  yaxt = "n"
)

axis(2, at = pretty(yI_lim))

legend(
  "topright",
  bty = "n",
  lwd = 2,
  legend = expression(
    paste(frac(1, t) * integral(I(r) * dr, 0, t))
  ),
  col = "blue"
)

## ---------- 3) R(t) ----------

yR_lim <- range(
  out_df2$avg_R_wnj,
  na.rm = TRUE
)

plot(
  out_df2$time,
  out_df2$avg_R_wnj,
  type = "l",
  lwd = 2,
  col = "blue",
  xlab = "",
  ylab = "",
  ylim = yR_lim,
  yaxt = "n"
)

axis(2, at = pretty(yR_lim))

legend(
  "bottomright",
  inset = c(0.01, 0.15),
  bty = "n",
  lwd = 2,
  legend = expression(
    paste(frac(1, t) * integral(R(r) * dr, 0, t))
  ),
  col = "blue"
)

mtext(
  "Time",
  side = 1,
  outer = TRUE,
  line = 1.5
)

mtext(
  "Time average of population",
  side = 2,
  outer = TRUE,
  line = 2.5
)