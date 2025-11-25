NAME = MHW
CC = cc
CFLAGS = -Wall -Wextra -Werror -pthread
INCLUDES = -I.
# 重要: 数学ライブラリ(-lm)とFFTW(-lfftw3)をリンク
LIBS = -lfftw3 -lm
RM = rm -f

SRCS = 2d_array_custom.c adv_utils.c adv.c main.c MHW_utils.c MHW.c

OBJS = $(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(INCLUDES) -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: all clean fclean re
