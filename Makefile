NAME = MHW
CC = cc
CFLAGS = -Wall -Wextra -Werror -pthread
INCLUDES = -I.
RM = rm -f

SRCS = 2d_array_custom.c adv_utils.c adv.c main.c MHW_utils.c MHW.c

OBJS = $(SRCS:.c=.o)

all: $(NAME)

$(NAME): $(OBJS)
		$(CC) $(CFLAGS) $(OBJS) $(INXLUDES) -o $@

%.o: %.c
		$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
		$(RM) $(OBJS)

fclean:	clean
		$(RM) $(NAME)

re: fclean all

.PHONY: all clean fclean re