library(keras)

adult_omitted =  adult %>% na.omit()
colSums(sapply(adult_omitted,is.na))
adult_omitted$target<- if(adult_omitted$target=='>50K'1)
adult_omitted$target<- if(adult_omitted$target=='<=50K'0)
    


spl<-sample.split(adult_omitted$target, SplitRatio=0.7)
train<-subset(adult_omitted,spl== TRUE)
test <-subset(adult_omitted, spl== FALSE)



model <- keras_model_sequential()

model %>% layer_dense(units = 64, activation = 'elu', 
                      input_shape = dim(train)[2]) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128, activation = 'elu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 256, activation = 'elu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128, activation = 'elu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = 'elu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(optimizer = 'adam', loss = c('binary_crossentropy'), metrics = c('accuracy'))

model

modelpath <- 'keras_dnn1.hdf5'
callback_list <- list(callback_early_stopping(patience = 5, monitor = 'val_loss'),
                      callback_model_checkpoint(filepath = modelpath, 
                                                save_best_only = TRUE))
start <- Sys.time()
history <- model %>% fit(as.matrix(train), train$target, batch_size = 64, 
                         epochs = 20, callbacks = callback_list, 
                         validation_split = 0.25)
end <- Sys.time()

dnn_elapsed <- as.numeric(difftime(end, start, units = 's'))


dnn_model <- load_model_hdf5(modelpath)
dnn_pred <- best_dnn_model %>% predict(as.matrix(test))

