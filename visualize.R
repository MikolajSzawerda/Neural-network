library(grid)
library(data.table)
library(gridExtra)
library(grid)
library(tidyr)
library(dplyr)
library(gt)
library(gtExtras)
library(ggplot2)
library(ggalt)
library(scales)
library(ggpubr)

exec_data <- fread("results/experiments.csv")

prepare_mse_plot <- function (row){
    mse_data <- fread(row['mse_path'])
    mse_data <- data.frame(mse_data)
    plt <-ggplot(mse_data, aes(x=it,y=mse, color="#0091ff")) +
      xlab("Liczba iteracji") +
      ylab("MSE") +
      ggtitle(row['name']) +
      scale_y_continuous(trans='log10',
                         breaks=trans_breaks('log10', function(x) 10^x),
                         labels=trans_format('log10', math_format(10^.x))) +
      geom_point(show.legend = FALSE)
    ggsave(sprintf("plots/%s_mse.png", row['path']), plt, width = 5, height = 5, units = "in")
  plt
}

prepare_predict_plot <- function(row){
  aprox_data <- fread(row['predict_path'])
  aprox_data <- data.frame(aprox_data)

  plt <- ggplot(aprox_data, aes(x=x)) +
    geom_line(aes(y=y_predict, colour="przewidywana")) +
    geom_line(aes(y=y, colour="dokładna")) +
    xlab("X") +
    ylab("Y") +
    ggtitle(row['name']) +
    scale_color_manual(name='Legenda', values=c('przewidywana'='red', 'dokładna'='blue'))
  ggsave(sprintf("plots/%s_predict.png", row['path']), plt, width = 5, height = 5, units = "in")
  plt
}

prepare_time_plot <- function (data) {
  plt <- ggplot(data, aes(x=name, y=time)) +
    xlab("Eksperyment") +
    ylab("Czas") +
    ggtitle("Wykres czasu wykonania") +
    scale_y_continuous(breaks = seq(0, max(exec_data$time), 0.25)) +
    geom_bar(stat='identity')
  ggsave("plots/execution_time.png", plt, width = 5, height = 5, units = "in")
}

apply(exec_data, 1, prepare_mse_plot)
apply(exec_data, 1, prepare_predict_plot)
prepare_time_plot(exec_data)
