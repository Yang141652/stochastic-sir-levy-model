library(MASS)
library(SymTS)

## ====== Parameters ======

Lambda  <- 8
mu      <- 5.3
beta    <- 4.8
epsilon <- 0.5
eta     <- 1

gamma_1 <- -0.3
gamma_2 <- 0.5
gamma_3 <- 0.2

## ====== Brownian covariance ======

L <- matrix(c(3, 2, 1,
              2, 3, 2,
              1, 2, 4),
            nrow = 3, byrow = TRUE)

L_1 <- 0.001 * L

## ====== Initial values and time ======

S0 <- 1.6
I0 <- 0.4
R0 <- 0.04

dt   <- 0.01
Tend <- 600
time <- seq(0, Tend, by = dt)
steps <- length(time)
n_inc <- steps - 1

n_paths <- 100

## ====== Jump parameters and directions ======

alpha <- 0.4
ell   <- 0.4
zeta  <- c(0.1, 0.1, 0.1)
c_j   <- c(1, 5, 5, 5, 1, 1)

w <- 0.5
theta_values <- 2 * pi * (1:6) / 6

direction_matrix <- rbind(
  sqrt(1 - w^2) * cos(theta_values),
  sqrt(1 - w^2) * sin(theta_values),
  rep(w, 6)
)

print(direction_matrix)

## ====== Storage for all paths ======

S_wn_all  <- matrix(0, steps, n_paths)
I_wn_all  <- matrix(0, steps, n_paths)
R_wn_all  <- matrix(0, steps, n_paths)

S_wnj_all <- matrix(0, steps, n_paths)
I_wnj_all <- matrix(0, steps, n_paths)
R_wnj_all <- matrix(0, steps, n_paths)

## ====== Run simulation paths ======

for (sim in seq_len(n_paths)) {

  cat("Simulation path:", sim, "of", n_paths, "\n")

  ## Six CTS increments per time step
  jump_matrix <- do.call(
    rbind,
    lapply(seq_len(6), function(j) {
      rCTS(n_inc, alpha = alpha, c = c_j[j] * dt,
           ell = ell, mu = 0)
    })
  )

  ## Apply the componentwise exponential map after forming
  ## the three-dimensional increment
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
  ## Correlated Brownian increments
  dW_matrix <- mvrnorm(
    n = n_inc,
    mu = c(0, 0, 0),
    Sigma = L_1 * dt
  )

  ## White-noise path
  S_wn <- I_wn <- R_wn <- numeric(steps)
  S_wn[1] <- S0
  I_wn[1] <- I0
  R_wn[1] <- R0

  ## White-noise plus jump path
  S_wnj <- I_wnj <- R_wnj <- numeric(steps)
  S_wnj[1] <- S0
  I_wnj[1] <- I0
  R_wnj[1] <- R0

  for (k in 2:steps) {

    p <- k - 1

    ## White noise
    S_wn[k] <- max(0, S_wn[p] +
      (Lambda - mu * S_wn[p] - beta * S_wn[p] * I_wn[p]) * dt +
      S_wn[p] * dW_matrix[p, 1])

    I_wn[k] <- max(0, I_wn[p] +
      (beta * S_wn[p] - mu - epsilon - eta) * I_wn[p] * dt +
      I_wn[p] * dW_matrix[p, 2])

    R_wn[k] <- max(0, R_wn[p] +
      (eta * I_wn[p] - mu * R_wn[p]) * dt +
      R_wn[p] * dW_matrix[p, 3])

    ## White noise + tempered stable increments
    S_wnj[k] <- max(0, S_wnj[p] +
      (Lambda - (mu - gamma_1) * S_wnj[p] -
         beta * S_wnj[p] * I_wnj[p]) * dt +
      S_wnj[p] * (dW_matrix[p, 1] + jump_vector[1, p]))

    I_wnj[k] <- max(0, I_wnj[p] +
      (beta * S_wnj[p] - mu - epsilon - eta + gamma_2) *
        I_wnj[p] * dt +
      I_wnj[p] * (dW_matrix[p, 2] + jump_vector[2, p]))

    R_wnj[k] <- max(0, R_wnj[p] +
      (eta * I_wnj[p] - (mu - gamma_3) * R_wnj[p]) * dt +
      R_wnj[p] * (dW_matrix[p, 3] + jump_vector[3, p]))
  }

  S_wn_all[, sim]  <- S_wn
  I_wn_all[, sim]  <- I_wn
  R_wn_all[, sim]  <- R_wn

  S_wnj_all[, sim] <- S_wnj
  I_wnj_all[, sim] <- I_wnj
  R_wnj_all[, sim] <- R_wnj
}

## ====== Mean trajectories ======

out_mean <- data.frame(
  time = time,
  S_wn_mean  = rowMeans(S_wn_all),
  I_wn_mean  = rowMeans(I_wn_all),
  R_wn_mean  = rowMeans(R_wn_all),
  S_wnj_mean = rowMeans(S_wnj_all),
  I_wnj_mean = rowMeans(I_wnj_all),
  R_wnj_mean = rowMeans(R_wnj_all)
)

## ====== Plot mean trajectories ======

par(mfrow = c(1, 1),
    mar = c(4, 4, 2, 1),
    oma = c(0, 0, 2, 0))

plot(out_mean$time, out_mean$S_wn_mean,
     type = "l", col = "blue", lwd = 2, lty = 3,
     xlab = "Time (dimensionless)",
     ylab = "Mean susceptible population S(t)",
     ylim = range(0, out_mean$S_wn_mean, out_mean$S_wnj_mean))

lines(out_mean$time, out_mean$S_wnj_mean,
      col = "red", lwd = 2, lty = 3)

legend("topright",
       legend = c("White noise: S(t)",
                  "White noise + Tempered stable: S(t)"),
       col = c("blue", "red"), lwd = 2, lty = 3,
       bty = "n", cex = 0.8)

plot(out_mean$time, out_mean$I_wn_mean,
     type = "l", col = "blue", lwd = 2, lty = 3,
     xlab = "Time (dimensionless)",
     ylab = "Mean infected population I(t)",
     ylim = range(0, out_mean$I_wn_mean, out_mean$I_wnj_mean))

lines(out_mean$time, out_mean$I_wnj_mean,
      col = "red", lwd = 2, lty = 3)

legend("topright",
       legend = c("White noise: I(t)",
                  "White noise + Tempered stable: I(t)"),
       col = c("blue", "red"), lwd = 2, lty = 3,
       bty = "n", cex = 0.8)

plot(out_mean$time, out_mean$R_wn_mean,
     type = "l", col = "blue", lwd = 2, lty = 3,
     xlab = "Time (dimensionless)",
     ylab = "Mean recovered population R(t)",
     ylim = range(0, out_mean$R_wn_mean, out_mean$R_wnj_mean))

lines(out_mean$time, out_mean$R_wnj_mean,
      col = "red", lwd = 2, lty = 3)

legend("topright",
       legend = c("White noise: R(t)",
                  "White noise + Tempered stable: R(t)"),
       col = c("blue", "red"), lwd = 2, lty = 3,
       bty = "n", cex = 0.8)