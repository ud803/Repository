## aes ##
## Aesthetic mappings
## aes(x, y, ...)
# Aesthetics supplied to ggplot() are used as defaults for every layer
# you can override them, or supply different aesthetics for each layer

ggplot(mpg, aes(displ, hwy)) + geom_point()
