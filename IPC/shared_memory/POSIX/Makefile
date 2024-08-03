CC		= gcc
CFLAGS	= -g
OBJS	= shared_memory.o
EXE		= creator writer reader

all: $(OBJS) $(EXE)

$(OBJS): shared_memory.c
	$(CC) $(CFLAGS) -c shared_memory.c -o $(OBJS)

$(EXE): %: %.c $(SHARED_OBJ)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@

clean: 
	rm -f *.o *.$(OBJS) $(EXE)
