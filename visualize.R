library(data.table)
library(ggplot2)
library(scales)
library(tidyr)
library(dplyr)
library(gt)
library(gtExtras)
theme_set(theme_classic())


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
    scale_color_manual(name='Legenda', values=c('przewidywana'='red', 'dokładna'='blue')) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
            plot.background = element_rect(fill = "#f7f7f7"),
            panel.background = element_rect(fill = "#f7f7f7"),
            panel.grid.minor = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.major.x = element_line(),
            axis.ticks = element_blank(),
            legend.position = "bottom",
            panel.border = element_blank())
  ggsave(sprintf("plots/%s_predict.png", row['path']), plt, width = 5, height = 5, units = "in")
  plt
}

prepare_time_plot <- function (data) {
  plt <- ggplot(data[order(data$name)], aes(x=name, y=time)) +
    xlab("Eksperyment") +
    ylab("Czas[s]") +
    ggtitle("Wykres czasu wykonania") +
    geom_bar(stat='identity')
  ggsave("plots/execution_time.png", plt, width = 8, height = 5, units = "in")
}

prepare_summary_table <- function (data) {
  data %>%
    select(name, time, mse) %>%
    gt() %>%
    fmt_number(
      columns = 2,
      decimals = 2,
      suffixing = TRUE
    ) %>%
    fmt_number(
      columns = 3,
      decimals = 5,
      suffixing = TRUE
    ) %>%
    cols_label(
      name = "Nazwa",
      time = "Czas wykonania",
      mse = "MSE"
    ) %>%
    tab_header("Podsumowanie wyników")
}

# apply(exec_data, 1, prepare_mse_plot)
# apply(exec_data, 1, prepare_predict_plot)
prepare_time_plot(exec_data)
# gtsave(prepare_summary_table(exec_data), "plots/summary_table.png")

