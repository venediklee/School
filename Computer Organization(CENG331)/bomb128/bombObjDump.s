
bomb:     file format elf64-x86-64


Disassembly of section .init:

0000000000001818 <_init>:
    1818:	48 83 ec 08          	sub    $0x8,%rsp
    181c:	48 8b 05 c5 37 20 00 	mov    0x2037c5(%rip),%rax        # 204fe8 <__gmon_start__>
    1823:	48 85 c0             	test   %rax,%rax
    1826:	74 02                	je     182a <_init+0x12>
    1828:	ff d0                	callq  *%rax
    182a:	48 83 c4 08          	add    $0x8,%rsp
    182e:	c3                   	retq   

Disassembly of section .plt:

0000000000001830 <.plt>:
    1830:	ff 35 b2 36 20 00    	pushq  0x2036b2(%rip)        # 204ee8 <_GLOBAL_OFFSET_TABLE_+0x8>
    1836:	ff 25 b4 36 20 00    	jmpq   *0x2036b4(%rip)        # 204ef0 <_GLOBAL_OFFSET_TABLE_+0x10>
    183c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001840 <__strcat_chk@plt>:
    1840:	ff 25 b2 36 20 00    	jmpq   *0x2036b2(%rip)        # 204ef8 <__strcat_chk@GLIBC_2.3.4>
    1846:	68 00 00 00 00       	pushq  $0x0
    184b:	e9 e0 ff ff ff       	jmpq   1830 <.plt>

0000000000001850 <getenv@plt>:
    1850:	ff 25 aa 36 20 00    	jmpq   *0x2036aa(%rip)        # 204f00 <getenv@GLIBC_2.2.5>
    1856:	68 01 00 00 00       	pushq  $0x1
    185b:	e9 d0 ff ff ff       	jmpq   1830 <.plt>

0000000000001860 <strcasecmp@plt>:
    1860:	ff 25 a2 36 20 00    	jmpq   *0x2036a2(%rip)        # 204f08 <strcasecmp@GLIBC_2.2.5>
    1866:	68 02 00 00 00       	pushq  $0x2
    186b:	e9 c0 ff ff ff       	jmpq   1830 <.plt>

0000000000001870 <__errno_location@plt>:
    1870:	ff 25 9a 36 20 00    	jmpq   *0x20369a(%rip)        # 204f10 <__errno_location@GLIBC_2.2.5>
    1876:	68 03 00 00 00       	pushq  $0x3
    187b:	e9 b0 ff ff ff       	jmpq   1830 <.plt>

0000000000001880 <strcpy@plt>:
    1880:	ff 25 92 36 20 00    	jmpq   *0x203692(%rip)        # 204f18 <strcpy@GLIBC_2.2.5>
    1886:	68 04 00 00 00       	pushq  $0x4
    188b:	e9 a0 ff ff ff       	jmpq   1830 <.plt>

0000000000001890 <puts@plt>:
    1890:	ff 25 8a 36 20 00    	jmpq   *0x20368a(%rip)        # 204f20 <puts@GLIBC_2.2.5>
    1896:	68 05 00 00 00       	pushq  $0x5
    189b:	e9 90 ff ff ff       	jmpq   1830 <.plt>

00000000000018a0 <write@plt>:
    18a0:	ff 25 82 36 20 00    	jmpq   *0x203682(%rip)        # 204f28 <write@GLIBC_2.2.5>
    18a6:	68 06 00 00 00       	pushq  $0x6
    18ab:	e9 80 ff ff ff       	jmpq   1830 <.plt>

00000000000018b0 <__stack_chk_fail@plt>:
    18b0:	ff 25 7a 36 20 00    	jmpq   *0x20367a(%rip)        # 204f30 <__stack_chk_fail@GLIBC_2.4>
    18b6:	68 07 00 00 00       	pushq  $0x7
    18bb:	e9 70 ff ff ff       	jmpq   1830 <.plt>

00000000000018c0 <alarm@plt>:
    18c0:	ff 25 72 36 20 00    	jmpq   *0x203672(%rip)        # 204f38 <alarm@GLIBC_2.2.5>
    18c6:	68 08 00 00 00       	pushq  $0x8
    18cb:	e9 60 ff ff ff       	jmpq   1830 <.plt>

00000000000018d0 <close@plt>:
    18d0:	ff 25 6a 36 20 00    	jmpq   *0x20366a(%rip)        # 204f40 <close@GLIBC_2.2.5>
    18d6:	68 09 00 00 00       	pushq  $0x9
    18db:	e9 50 ff ff ff       	jmpq   1830 <.plt>

00000000000018e0 <read@plt>:
    18e0:	ff 25 62 36 20 00    	jmpq   *0x203662(%rip)        # 204f48 <read@GLIBC_2.2.5>
    18e6:	68 0a 00 00 00       	pushq  $0xa
    18eb:	e9 40 ff ff ff       	jmpq   1830 <.plt>

00000000000018f0 <fgets@plt>:
    18f0:	ff 25 5a 36 20 00    	jmpq   *0x20365a(%rip)        # 204f50 <fgets@GLIBC_2.2.5>
    18f6:	68 0b 00 00 00       	pushq  $0xb
    18fb:	e9 30 ff ff ff       	jmpq   1830 <.plt>

0000000000001900 <signal@plt>:
    1900:	ff 25 52 36 20 00    	jmpq   *0x203652(%rip)        # 204f58 <signal@GLIBC_2.2.5>
    1906:	68 0c 00 00 00       	pushq  $0xc
    190b:	e9 20 ff ff ff       	jmpq   1830 <.plt>

0000000000001910 <gethostbyname@plt>:
    1910:	ff 25 4a 36 20 00    	jmpq   *0x20364a(%rip)        # 204f60 <gethostbyname@GLIBC_2.2.5>
    1916:	68 0d 00 00 00       	pushq  $0xd
    191b:	e9 10 ff ff ff       	jmpq   1830 <.plt>

0000000000001920 <__memmove_chk@plt>:
    1920:	ff 25 42 36 20 00    	jmpq   *0x203642(%rip)        # 204f68 <__memmove_chk@GLIBC_2.3.4>
    1926:	68 0e 00 00 00       	pushq  $0xe
    192b:	e9 00 ff ff ff       	jmpq   1830 <.plt>

0000000000001930 <strtol@plt>:
    1930:	ff 25 3a 36 20 00    	jmpq   *0x20363a(%rip)        # 204f70 <strtol@GLIBC_2.2.5>
    1936:	68 0f 00 00 00       	pushq  $0xf
    193b:	e9 f0 fe ff ff       	jmpq   1830 <.plt>

0000000000001940 <fflush@plt>:
    1940:	ff 25 32 36 20 00    	jmpq   *0x203632(%rip)        # 204f78 <fflush@GLIBC_2.2.5>
    1946:	68 10 00 00 00       	pushq  $0x10
    194b:	e9 e0 fe ff ff       	jmpq   1830 <.plt>

0000000000001950 <__isoc99_sscanf@plt>:
    1950:	ff 25 2a 36 20 00    	jmpq   *0x20362a(%rip)        # 204f80 <__isoc99_sscanf@GLIBC_2.7>
    1956:	68 11 00 00 00       	pushq  $0x11
    195b:	e9 d0 fe ff ff       	jmpq   1830 <.plt>

0000000000001960 <__printf_chk@plt>:
    1960:	ff 25 22 36 20 00    	jmpq   *0x203622(%rip)        # 204f88 <__printf_chk@GLIBC_2.3.4>
    1966:	68 12 00 00 00       	pushq  $0x12
    196b:	e9 c0 fe ff ff       	jmpq   1830 <.plt>

0000000000001970 <fopen@plt>:
    1970:	ff 25 1a 36 20 00    	jmpq   *0x20361a(%rip)        # 204f90 <fopen@GLIBC_2.2.5>
    1976:	68 13 00 00 00       	pushq  $0x13
    197b:	e9 b0 fe ff ff       	jmpq   1830 <.plt>

0000000000001980 <gethostname@plt>:
    1980:	ff 25 12 36 20 00    	jmpq   *0x203612(%rip)        # 204f98 <gethostname@GLIBC_2.2.5>
    1986:	68 14 00 00 00       	pushq  $0x14
    198b:	e9 a0 fe ff ff       	jmpq   1830 <.plt>

0000000000001990 <exit@plt>:
    1990:	ff 25 0a 36 20 00    	jmpq   *0x20360a(%rip)        # 204fa0 <exit@GLIBC_2.2.5>
    1996:	68 15 00 00 00       	pushq  $0x15
    199b:	e9 90 fe ff ff       	jmpq   1830 <.plt>

00000000000019a0 <connect@plt>:
    19a0:	ff 25 02 36 20 00    	jmpq   *0x203602(%rip)        # 204fa8 <connect@GLIBC_2.2.5>
    19a6:	68 16 00 00 00       	pushq  $0x16
    19ab:	e9 80 fe ff ff       	jmpq   1830 <.plt>

00000000000019b0 <__fprintf_chk@plt>:
    19b0:	ff 25 fa 35 20 00    	jmpq   *0x2035fa(%rip)        # 204fb0 <__fprintf_chk@GLIBC_2.3.4>
    19b6:	68 17 00 00 00       	pushq  $0x17
    19bb:	e9 70 fe ff ff       	jmpq   1830 <.plt>

00000000000019c0 <sleep@plt>:
    19c0:	ff 25 f2 35 20 00    	jmpq   *0x2035f2(%rip)        # 204fb8 <sleep@GLIBC_2.2.5>
    19c6:	68 18 00 00 00       	pushq  $0x18
    19cb:	e9 60 fe ff ff       	jmpq   1830 <.plt>

00000000000019d0 <__ctype_b_loc@plt>:
    19d0:	ff 25 ea 35 20 00    	jmpq   *0x2035ea(%rip)        # 204fc0 <__ctype_b_loc@GLIBC_2.3>
    19d6:	68 19 00 00 00       	pushq  $0x19
    19db:	e9 50 fe ff ff       	jmpq   1830 <.plt>

00000000000019e0 <__sprintf_chk@plt>:
    19e0:	ff 25 e2 35 20 00    	jmpq   *0x2035e2(%rip)        # 204fc8 <__sprintf_chk@GLIBC_2.3.4>
    19e6:	68 1a 00 00 00       	pushq  $0x1a
    19eb:	e9 40 fe ff ff       	jmpq   1830 <.plt>

00000000000019f0 <socket@plt>:
    19f0:	ff 25 da 35 20 00    	jmpq   *0x2035da(%rip)        # 204fd0 <socket@GLIBC_2.2.5>
    19f6:	68 1b 00 00 00       	pushq  $0x1b
    19fb:	e9 30 fe ff ff       	jmpq   1830 <.plt>

Disassembly of section .plt.got:

0000000000001a00 <__cxa_finalize@plt>:
    1a00:	ff 25 f2 35 20 00    	jmpq   *0x2035f2(%rip)        # 204ff8 <__cxa_finalize@GLIBC_2.2.5>
    1a06:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000001a10 <_start>:
    1a10:	31 ed                	xor    %ebp,%ebp
    1a12:	49 89 d1             	mov    %rdx,%r9
    1a15:	5e                   	pop    %rsi
    1a16:	48 89 e2             	mov    %rsp,%rdx
    1a19:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    1a1d:	50                   	push   %rax
    1a1e:	54                   	push   %rsp
    1a1f:	4c 8d 05 5a 1a 00 00 	lea    0x1a5a(%rip),%r8        # 3480 <__libc_csu_fini>
    1a26:	48 8d 0d e3 19 00 00 	lea    0x19e3(%rip),%rcx        # 3410 <__libc_csu_init>
    1a2d:	48 8d 3d e6 00 00 00 	lea    0xe6(%rip),%rdi        # 1b1a <main>
    1a34:	ff 15 a6 35 20 00    	callq  *0x2035a6(%rip)        # 204fe0 <__libc_start_main@GLIBC_2.2.5>
    1a3a:	f4                   	hlt    
    1a3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001a40 <deregister_tm_clones>:
    1a40:	48 8d 3d 39 3c 20 00 	lea    0x203c39(%rip),%rdi        # 205680 <stdout@@GLIBC_2.2.5>
    1a47:	55                   	push   %rbp
    1a48:	48 8d 05 31 3c 20 00 	lea    0x203c31(%rip),%rax        # 205680 <stdout@@GLIBC_2.2.5>
    1a4f:	48 39 f8             	cmp    %rdi,%rax
    1a52:	48 89 e5             	mov    %rsp,%rbp
    1a55:	74 19                	je     1a70 <deregister_tm_clones+0x30>
    1a57:	48 8b 05 7a 35 20 00 	mov    0x20357a(%rip),%rax        # 204fd8 <_ITM_deregisterTMCloneTable>
    1a5e:	48 85 c0             	test   %rax,%rax
    1a61:	74 0d                	je     1a70 <deregister_tm_clones+0x30>
    1a63:	5d                   	pop    %rbp
    1a64:	ff e0                	jmpq   *%rax
    1a66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1a6d:	00 00 00 
    1a70:	5d                   	pop    %rbp
    1a71:	c3                   	retq   
    1a72:	0f 1f 40 00          	nopl   0x0(%rax)
    1a76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1a7d:	00 00 00 

0000000000001a80 <register_tm_clones>:
    1a80:	48 8d 3d f9 3b 20 00 	lea    0x203bf9(%rip),%rdi        # 205680 <stdout@@GLIBC_2.2.5>
    1a87:	48 8d 35 f2 3b 20 00 	lea    0x203bf2(%rip),%rsi        # 205680 <stdout@@GLIBC_2.2.5>
    1a8e:	55                   	push   %rbp
    1a8f:	48 29 fe             	sub    %rdi,%rsi
    1a92:	48 89 e5             	mov    %rsp,%rbp
    1a95:	48 c1 fe 03          	sar    $0x3,%rsi
    1a99:	48 89 f0             	mov    %rsi,%rax
    1a9c:	48 c1 e8 3f          	shr    $0x3f,%rax
    1aa0:	48 01 c6             	add    %rax,%rsi
    1aa3:	48 d1 fe             	sar    %rsi
    1aa6:	74 18                	je     1ac0 <register_tm_clones+0x40>
    1aa8:	48 8b 05 41 35 20 00 	mov    0x203541(%rip),%rax        # 204ff0 <_ITM_registerTMCloneTable>
    1aaf:	48 85 c0             	test   %rax,%rax
    1ab2:	74 0c                	je     1ac0 <register_tm_clones+0x40>
    1ab4:	5d                   	pop    %rbp
    1ab5:	ff e0                	jmpq   *%rax
    1ab7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    1abe:	00 00 
    1ac0:	5d                   	pop    %rbp
    1ac1:	c3                   	retq   
    1ac2:	0f 1f 40 00          	nopl   0x0(%rax)
    1ac6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1acd:	00 00 00 

0000000000001ad0 <__do_global_dtors_aux>:
    1ad0:	80 3d d1 3b 20 00 00 	cmpb   $0x0,0x203bd1(%rip)        # 2056a8 <completed.7696>
    1ad7:	75 2f                	jne    1b08 <__do_global_dtors_aux+0x38>
    1ad9:	48 83 3d 17 35 20 00 	cmpq   $0x0,0x203517(%rip)        # 204ff8 <__cxa_finalize@GLIBC_2.2.5>
    1ae0:	00 
    1ae1:	55                   	push   %rbp
    1ae2:	48 89 e5             	mov    %rsp,%rbp
    1ae5:	74 0c                	je     1af3 <__do_global_dtors_aux+0x23>
    1ae7:	48 8b 3d 1a 35 20 00 	mov    0x20351a(%rip),%rdi        # 205008 <__dso_handle>
    1aee:	e8 0d ff ff ff       	callq  1a00 <__cxa_finalize@plt>
    1af3:	e8 48 ff ff ff       	callq  1a40 <deregister_tm_clones>
    1af8:	c6 05 a9 3b 20 00 01 	movb   $0x1,0x203ba9(%rip)        # 2056a8 <completed.7696>
    1aff:	5d                   	pop    %rbp
    1b00:	c3                   	retq   
    1b01:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1b08:	f3 c3                	repz retq 
    1b0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001b10 <frame_dummy>:
    1b10:	55                   	push   %rbp
    1b11:	48 89 e5             	mov    %rsp,%rbp
    1b14:	5d                   	pop    %rbp
    1b15:	e9 66 ff ff ff       	jmpq   1a80 <register_tm_clones>

0000000000001b1a <main>:
    1b1a:	53                   	push   %rbx
    1b1b:	83 ff 01             	cmp    $0x1,%edi
    1b1e:	0f 84 f8 00 00 00    	je     1c1c <main+0x102>
    1b24:	48 89 f3             	mov    %rsi,%rbx
    1b27:	83 ff 02             	cmp    $0x2,%edi
    1b2a:	0f 85 21 01 00 00    	jne    1c51 <main+0x137>
    1b30:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
    1b34:	48 8d 35 8a 22 00 00 	lea    0x228a(%rip),%rsi        # 3dc5 <array.3433+0x745>
    1b3b:	e8 30 fe ff ff       	callq  1970 <fopen@plt>
    1b40:	48 89 05 69 3b 20 00 	mov    %rax,0x203b69(%rip)        # 2056b0 <infile>
    1b47:	48 85 c0             	test   %rax,%rax
    1b4a:	0f 84 df 00 00 00    	je     1c2f <main+0x115>
    1b50:	e8 b6 07 00 00       	callq  230b <initialize_bomb>
    1b55:	48 8d 3d cc 19 00 00 	lea    0x19cc(%rip),%rdi        # 3528 <_IO_stdin_used+0x88>
    1b5c:	e8 2f fd ff ff       	callq  1890 <puts@plt>
    1b61:	48 8d 3d 00 1a 00 00 	lea    0x1a00(%rip),%rdi        # 3568 <_IO_stdin_used+0xc8>
    1b68:	e8 23 fd ff ff       	callq  1890 <puts@plt>
    1b6d:	e8 b3 0a 00 00       	callq  2625 <read_line>
    1b72:	48 89 c7             	mov    %rax,%rdi
    1b75:	e8 fa 00 00 00       	callq  1c74 <phase_1>
    1b7a:	e8 ea 0b 00 00       	callq  2769 <phase_defused>
    1b7f:	48 8d 3d 12 1a 00 00 	lea    0x1a12(%rip),%rdi        # 3598 <_IO_stdin_used+0xf8>
    1b86:	e8 05 fd ff ff       	callq  1890 <puts@plt>
    1b8b:	e8 95 0a 00 00       	callq  2625 <read_line>
    1b90:	48 89 c7             	mov    %rax,%rdi
    1b93:	e8 9e 01 00 00       	callq  1d36 <phase_2>
    1b98:	e8 cc 0b 00 00       	callq  2769 <phase_defused>
    1b9d:	48 8d 3d 37 19 00 00 	lea    0x1937(%rip),%rdi        # 34db <_IO_stdin_used+0x3b>
    1ba4:	e8 e7 fc ff ff       	callq  1890 <puts@plt>
    1ba9:	e8 77 0a 00 00       	callq  2625 <read_line>
    1bae:	48 89 c7             	mov    %rax,%rdi
    1bb1:	e8 e8 01 00 00       	callq  1d9e <phase_3>
    1bb6:	e8 ae 0b 00 00       	callq  2769 <phase_defused>
    1bbb:	48 8d 3d 37 19 00 00 	lea    0x1937(%rip),%rdi        # 34f9 <_IO_stdin_used+0x59>
    1bc2:	e8 c9 fc ff ff       	callq  1890 <puts@plt>
    1bc7:	e8 59 0a 00 00       	callq  2625 <read_line>
    1bcc:	48 89 c7             	mov    %rax,%rdi
    1bcf:	e8 b9 02 00 00       	callq  1e8d <phase_4>
    1bd4:	e8 90 0b 00 00       	callq  2769 <phase_defused>
    1bd9:	48 8d 3d e8 19 00 00 	lea    0x19e8(%rip),%rdi        # 35c8 <_IO_stdin_used+0x128>
    1be0:	e8 ab fc ff ff       	callq  1890 <puts@plt>
    1be5:	e8 3b 0a 00 00       	callq  2625 <read_line>
    1bea:	48 89 c7             	mov    %rax,%rdi
    1bed:	e8 16 03 00 00       	callq  1f08 <phase_5>
    1bf2:	e8 72 0b 00 00       	callq  2769 <phase_defused>
    1bf7:	48 8d 3d 0a 19 00 00 	lea    0x190a(%rip),%rdi        # 3508 <_IO_stdin_used+0x68>
    1bfe:	e8 8d fc ff ff       	callq  1890 <puts@plt>
    1c03:	e8 1d 0a 00 00       	callq  2625 <read_line>
    1c08:	48 89 c7             	mov    %rax,%rdi
    1c0b:	e8 87 03 00 00       	callq  1f97 <phase_6>
    1c10:	e8 54 0b 00 00       	callq  2769 <phase_defused>
    1c15:	b8 00 00 00 00       	mov    $0x0,%eax
    1c1a:	5b                   	pop    %rbx
    1c1b:	c3                   	retq   
    1c1c:	48 8b 05 6d 3a 20 00 	mov    0x203a6d(%rip),%rax        # 205690 <stdin@@GLIBC_2.2.5>
    1c23:	48 89 05 86 3a 20 00 	mov    %rax,0x203a86(%rip)        # 2056b0 <infile>
    1c2a:	e9 21 ff ff ff       	jmpq   1b50 <main+0x36>
    1c2f:	48 8b 4b 08          	mov    0x8(%rbx),%rcx
    1c33:	48 8b 13             	mov    (%rbx),%rdx
    1c36:	48 8d 35 67 18 00 00 	lea    0x1867(%rip),%rsi        # 34a4 <_IO_stdin_used+0x4>
    1c3d:	bf 01 00 00 00       	mov    $0x1,%edi
    1c42:	e8 19 fd ff ff       	callq  1960 <__printf_chk@plt>
    1c47:	bf 08 00 00 00       	mov    $0x8,%edi
    1c4c:	e8 3f fd ff ff       	callq  1990 <exit@plt>
    1c51:	48 8b 16             	mov    (%rsi),%rdx
    1c54:	48 8d 35 66 18 00 00 	lea    0x1866(%rip),%rsi        # 34c1 <_IO_stdin_used+0x21>
    1c5b:	bf 01 00 00 00       	mov    $0x1,%edi
    1c60:	b8 00 00 00 00       	mov    $0x0,%eax
    1c65:	e8 f6 fc ff ff       	callq  1960 <__printf_chk@plt>
    1c6a:	bf 08 00 00 00       	mov    $0x8,%edi
    1c6f:	e8 1c fd ff ff       	callq  1990 <exit@plt>

0000000000001c74 <phase_1>:
    1c74:	55                   	push   %rbp
    1c75:	53                   	push   %rbx
    1c76:	48 83 ec 68          	sub    $0x68,%rsp
    1c7a:	48 89 fd             	mov    %rdi,%rbp
    1c7d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1c84:	00 00 
    1c86:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    1c8b:	31 c0                	xor    %eax,%eax
    1c8d:	48 b8 54 68 65 20 74 	movabs $0x7478657420656854,%rax
    1c94:	65 78 74 
    1c97:	ba 20 69 73 20       	mov    $0x20736920,%edx
    1c9c:	48 89 04 24          	mov    %rax,(%rsp)
    1ca0:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    1ca5:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    1cac:	00 00 
    1cae:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
    1cb5:	00 00 
    1cb7:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    1cbe:	00 00 
    1cc0:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
    1cc7:	00 00 
    1cc9:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
    1cd0:	00 00 
    1cd2:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    1cd9:	00 00 
    1cdb:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
    1ce2:	00 00 
    1ce4:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    1ceb:	00 00 
    1ced:	48 89 e3             	mov    %rsp,%rbx
    1cf0:	ba 50 00 00 00       	mov    $0x50,%edx
    1cf5:	48 8d 35 f4 18 00 00 	lea    0x18f4(%rip),%rsi        # 35f0 <_IO_stdin_used+0x150>
    1cfc:	48 89 df             	mov    %rbx,%rdi
    1cff:	e8 3c fb ff ff       	callq  1840 <__strcat_chk@plt>
    1d04:	48 89 de             	mov    %rbx,%rsi
    1d07:	48 89 ef             	mov    %rbp,%rdi
    1d0a:	e8 79 05 00 00       	callq  2288 <strings_not_equal>
    1d0f:	85 c0                	test   %eax,%eax
    1d11:	75 17                	jne    1d2a <phase_1+0xb6>
    1d13:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    1d18:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1d1f:	00 00 
    1d21:	75 0e                	jne    1d31 <phase_1+0xbd>
    1d23:	48 83 c4 68          	add    $0x68,%rsp
    1d27:	5b                   	pop    %rbx
    1d28:	5d                   	pop    %rbp
    1d29:	c3                   	retq   
    1d2a:	e8 79 08 00 00       	callq  25a8 <explode_bomb>
    1d2f:	eb e2                	jmp    1d13 <phase_1+0x9f>
    1d31:	e8 7a fb ff ff       	callq  18b0 <__stack_chk_fail@plt>

0000000000001d36 <phase_2>:
    1d36:	55                   	push   %rbp
    1d37:	53                   	push   %rbx
    1d38:	48 83 ec 28          	sub    $0x28,%rsp	
    1d3c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1d43:	00 00 
    1d45:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    1d4a:	31 c0                	xor    %eax,%eax
    1d4c:	48 89 e6             	mov    %rsp,%rsi
    1d4f:	e8 90 08 00 00       	callq  25e4 <read_six_numbers> # starts here
    1d54:	bb 02 00 00 00       	mov    $0x2,%ebx
    1d59:	48 89 e5             	mov    %rsp,%rbp
    1d5c:	eb 0a                	jmp    1d68 <phase_2+0x32>
    1d5e:	48 83 c3 01          	add    $0x1,%rbx
    1d62:	48 83 fb 06          	cmp    $0x6,%rbx
    1d66:	74 1a                	je     1d82 <phase_2+0x4c>
    1d68:	89 d8                	mov    %ebx,%eax
    1d6a:	c1 e8 1f             	shr    $0x1f,%eax
    1d6d:	01 d8                	add    %ebx,%eax
    1d6f:	d1 f8                	sar    %eax
    1d71:	03 44 9d fc          	add    -0x4(%rbp,%rbx,4),%eax
    1d75:	39 44 9d 00          	cmp    %eax,0x0(%rbp,%rbx,4)
    1d79:	74 e3                	je     1d5e <phase_2+0x28>
    1d7b:	e8 28 08 00 00       	callq  25a8 <explode_bomb>
    1d80:	eb dc                	jmp    1d5e <phase_2+0x28>
    1d82:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    1d87:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1d8e:	00 00 
    1d90:	75 07                	jne    1d99 <phase_2+0x63>
    1d92:	48 83 c4 28          	add    $0x28,%rsp
    1d96:	5b                   	pop    %rbx
    1d97:	5d                   	pop    %rbp
    1d98:	c3                   	retq   
    1d99:	e8 12 fb ff ff       	callq  18b0 <__stack_chk_fail@plt>

0000000000001d9e <phase_3>:
    1d9e:	48 83 ec 28          	sub    $0x28,%rsp
    1da2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1da9:	00 00 
    1dab:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    1db0:	31 c0                	xor    %eax,%eax
    1db2:	48 8d 4c 24 10       	lea    0x10(%rsp),%rcx
    1db7:	48 8d 54 24 0c       	lea    0xc(%rsp),%rdx
    1dbc:	4c 8d 44 24 14       	lea    0x14(%rsp),%r8
    1dc1:	48 8d 35 42 1b 00 00 	lea    0x1b42(%rip),%rsi        # 390a <array.3433+0x28a>
    1dc8:	e8 83 fb ff ff       	callq  1950 <__isoc99_sscanf@plt>
    1dcd:	83 f8 02             	cmp    $0x2,%eax
    1dd0:	7e 1b                	jle    1ded <phase_3+0x4f>
    1dd2:	83 7c 24 0c 07       	cmpl   $0x7,0xc(%rsp)
    1dd7:	77 4c                	ja     1e25 <phase_3+0x87>
    1dd9:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    1ddd:	48 8d 15 7c 18 00 00 	lea    0x187c(%rip),%rdx        # 3660 <_IO_stdin_used+0x1c0>
    1de4:	48 63 04 82          	movslq (%rdx,%rax,4),%rax
    1de8:	48 01 d0             	add    %rdx,%rax
    1deb:	ff e0                	jmpq   *%rax
    1ded:	e8 b6 07 00 00       	callq  25a8 <explode_bomb>
    1df2:	eb de                	jmp    1dd2 <phase_3+0x34>
    1df4:	ba 83 02 00 00       	mov    $0x283,%edx
    1df9:	eb 3b                	jmp    1e36 <phase_3+0x98>
    1dfb:	ba 80 03 00 00       	mov    $0x380,%edx
    1e00:	eb 34                	jmp    1e36 <phase_3+0x98>
    1e02:	ba 9e 00 00 00       	mov    $0x9e,%edx
    1e07:	eb 2d                	jmp    1e36 <phase_3+0x98>
    1e09:	ba 78 03 00 00       	mov    $0x378,%edx
    1e0e:	eb 26                	jmp    1e36 <phase_3+0x98>
    1e10:	ba 79 02 00 00       	mov    $0x279,%edx
    1e15:	eb 1f                	jmp    1e36 <phase_3+0x98>
    1e17:	ba d9 00 00 00       	mov    $0xd9,%edx
    1e1c:	eb 18                	jmp    1e36 <phase_3+0x98>
    1e1e:	ba 5d 00 00 00       	mov    $0x5d,%edx
    1e23:	eb 11                	jmp    1e36 <phase_3+0x98>
    1e25:	e8 7e 07 00 00       	callq  25a8 <explode_bomb>
    1e2a:	ba 00 00 00 00       	mov    $0x0,%edx
    1e2f:	eb 05                	jmp    1e36 <phase_3+0x98>
    1e31:	ba bc 00 00 00       	mov    $0xbc,%edx
    1e36:	8b 74 24 14          	mov    0x14(%rsp),%esi
    1e3a:	8b 7c 24 10          	mov    0x10(%rsp),%edi
    1e3e:	e8 d2 03 00 00       	callq  2215 <check_division_equal>
    1e43:	85 c0                	test   %eax,%eax
    1e45:	74 15                	je     1e5c <phase_3+0xbe>
    1e47:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    1e4c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1e53:	00 00 
    1e55:	75 0c                	jne    1e63 <phase_3+0xc5>
    1e57:	48 83 c4 28          	add    $0x28,%rsp
    1e5b:	c3                   	retq   
    1e5c:	e8 47 07 00 00       	callq  25a8 <explode_bomb>
    1e61:	eb e4                	jmp    1e47 <phase_3+0xa9>
    1e63:	e8 48 fa ff ff       	callq  18b0 <__stack_chk_fail@plt>

0000000000001e68 <func4>:
    1e68:	83 ff 01             	cmp    $0x1,%edi
    1e6b:	7f 0c                	jg     1e79 <func4+0x11>
    1e6d:	c7 06 01 00 00 00    	movl   $0x1,(%rsi)
    1e73:	b8 01 00 00 00       	mov    $0x1,%eax
    1e78:	c3                   	retq   
    1e79:	53                   	push   %rbx
    1e7a:	48 89 f3             	mov    %rsi,%rbx
    1e7d:	83 ef 01             	sub    $0x1,%edi
    1e80:	e8 e3 ff ff ff       	callq  1e68 <func4>
    1e85:	89 c2                	mov    %eax,%edx
    1e87:	03 03                	add    (%rbx),%eax
    1e89:	89 13                	mov    %edx,(%rbx)
    1e8b:	5b                   	pop    %rbx
    1e8c:	c3                   	retq   

0000000000001e8d <phase_4>:
    1e8d:	48 83 ec 28          	sub    $0x28,%rsp
    1e91:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1e98:	00 00 
    1e9a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    1e9f:	31 c0                	xor    %eax,%eax
    1ea1:	48 8d 4c 24 10       	lea    0x10(%rsp),%rcx
    1ea6:	48 8d 54 24 0c       	lea    0xc(%rsp),%rdx
    1eab:	48 8d 35 5b 1a 00 00 	lea    0x1a5b(%rip),%rsi        # 390d <array.3433+0x28d>
    1eb2:	e8 99 fa ff ff       	callq  1950 <__isoc99_sscanf@plt>
    1eb7:	83 f8 02             	cmp    $0x2,%eax
    1eba:	75 0c                	jne    1ec8 <phase_4+0x3b>
    1ebc:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    1ec0:	83 e8 01             	sub    $0x1,%eax
    1ec3:	83 f8 13             	cmp    $0x13,%eax
    1ec6:	76 05                	jbe    1ecd <phase_4+0x40>
    1ec8:	e8 db 06 00 00       	callq  25a8 <explode_bomb>
    1ecd:	48 8d 74 24 14       	lea    0x14(%rsp),%rsi
    1ed2:	8b 7c 24 0c          	mov    0xc(%rsp),%edi
    1ed6:	e8 8d ff ff ff       	callq  1e68 <func4>
    1edb:	83 7c 24 14 05       	cmpl   $0x5,0x14(%rsp)
    1ee0:	75 07                	jne    1ee9 <phase_4+0x5c>
    1ee2:	83 7c 24 10 05       	cmpl   $0x5,0x10(%rsp)
    1ee7:	74 05                	je     1eee <phase_4+0x61>
    1ee9:	e8 ba 06 00 00       	callq  25a8 <explode_bomb>
    1eee:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    1ef3:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1efa:	00 00 
    1efc:	75 05                	jne    1f03 <phase_4+0x76>
    1efe:	48 83 c4 28          	add    $0x28,%rsp
    1f02:	c3                   	retq   
    1f03:	e8 a8 f9 ff ff       	callq  18b0 <__stack_chk_fail@plt>

0000000000001f08 <phase_5>:
    1f08:	48 83 ec 18          	sub    $0x18,%rsp
    1f0c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1f13:	00 00 
    1f15:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    1f1a:	31 c0                	xor    %eax,%eax
    1f1c:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
    1f21:	48 89 e2             	mov    %rsp,%rdx
    1f24:	48 8d 35 e2 19 00 00 	lea    0x19e2(%rip),%rsi        # 390d <array.3433+0x28d>
    1f2b:	e8 20 fa ff ff       	callq  1950 <__isoc99_sscanf@plt>
    1f30:	83 f8 01             	cmp    $0x1,%eax
    1f33:	7e 56                	jle    1f8b <phase_5+0x83>
    1f35:	8b 34 24             	mov    (%rsp),%esi
    1f38:	83 e6 0f             	and    $0xf,%esi
    1f3b:	89 34 24             	mov    %esi,(%rsp)
    1f3e:	83 fe 0f             	cmp    $0xf,%esi
    1f41:	74 2e                	je     1f71 <phase_5+0x69>
    1f43:	89 f0                	mov    %esi,%eax
    1f45:	ba 00 00 00 00       	mov    $0x0,%edx
    1f4a:	48 8d 3d 2f 17 00 00 	lea    0x172f(%rip),%rdi        # 3680 <array.3433>
    1f51:	48 63 c8             	movslq %eax,%rcx
    1f54:	03 14 8f             	add    (%rdi,%rcx,4),%edx
    1f57:	83 c0 01             	add    $0x1,%eax
    1f5a:	83 f8 0f             	cmp    $0xf,%eax
    1f5d:	75 f2                	jne    1f51 <phase_5+0x49>
    1f5f:	c7 04 24 0f 00 00 00 	movl   $0xf,(%rsp)
    1f66:	83 fe 09             	cmp    $0x9,%esi
    1f69:	75 06                	jne    1f71 <phase_5+0x69>
    1f6b:	39 54 24 04          	cmp    %edx,0x4(%rsp)
    1f6f:	74 05                	je     1f76 <phase_5+0x6e>
    1f71:	e8 32 06 00 00       	callq  25a8 <explode_bomb>
    1f76:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    1f7b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1f82:	00 00 
    1f84:	75 0c                	jne    1f92 <phase_5+0x8a>
    1f86:	48 83 c4 18          	add    $0x18,%rsp
    1f8a:	c3                   	retq   
    1f8b:	e8 18 06 00 00       	callq  25a8 <explode_bomb>
    1f90:	eb a3                	jmp    1f35 <phase_5+0x2d>
    1f92:	e8 19 f9 ff ff       	callq  18b0 <__stack_chk_fail@plt>

0000000000001f97 <phase_6>:
    1f97:	41 55                	push   %r13
    1f99:	41 54                	push   %r12
    1f9b:	55                   	push   %rbp
    1f9c:	53                   	push   %rbx
    1f9d:	48 83 ec 68          	sub    $0x68,%rsp
    1fa1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1fa8:	00 00 
    1faa:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    1faf:	31 c0                	xor    %eax,%eax
    1fb1:	49 89 e4             	mov    %rsp,%r12
    1fb4:	4c 89 e6             	mov    %r12,%rsi
    1fb7:	e8 28 06 00 00       	callq  25e4 <read_six_numbers>
    1fbc:	41 bd 00 00 00 00    	mov    $0x0,%r13d
    1fc2:	eb 25                	jmp    1fe9 <phase_6+0x52>
    1fc4:	e8 df 05 00 00       	callq  25a8 <explode_bomb>
    1fc9:	eb 2d                	jmp    1ff8 <phase_6+0x61>
    1fcb:	83 c3 01             	add    $0x1,%ebx
    1fce:	83 fb 05             	cmp    $0x5,%ebx
    1fd1:	7f 12                	jg     1fe5 <phase_6+0x4e>
    1fd3:	48 63 c3             	movslq %ebx,%rax
    1fd6:	8b 04 84             	mov    (%rsp,%rax,4),%eax
    1fd9:	39 45 00             	cmp    %eax,0x0(%rbp)
    1fdc:	75 ed                	jne    1fcb <phase_6+0x34>
    1fde:	e8 c5 05 00 00       	callq  25a8 <explode_bomb>
    1fe3:	eb e6                	jmp    1fcb <phase_6+0x34>
    1fe5:	49 83 c4 04          	add    $0x4,%r12
    1fe9:	4c 89 e5             	mov    %r12,%rbp
    1fec:	41 8b 04 24          	mov    (%r12),%eax
    1ff0:	83 e8 01             	sub    $0x1,%eax
    1ff3:	83 f8 05             	cmp    $0x5,%eax
    1ff6:	77 cc                	ja     1fc4 <phase_6+0x2d>
    1ff8:	41 83 c5 01          	add    $0x1,%r13d
    1ffc:	41 83 fd 06          	cmp    $0x6,%r13d
    2000:	74 35                	je     2037 <phase_6+0xa0>
    2002:	44 89 eb             	mov    %r13d,%ebx
    2005:	eb cc                	jmp    1fd3 <phase_6+0x3c>
    2007:	48 8b 52 08          	mov    0x8(%rdx),%rdx
    200b:	83 c0 01             	add    $0x1,%eax
    200e:	39 c8                	cmp    %ecx,%eax
    2010:	75 f5                	jne    2007 <phase_6+0x70>
    2012:	48 89 54 f4 20       	mov    %rdx,0x20(%rsp,%rsi,8)
    2017:	48 83 c6 01          	add    $0x1,%rsi
    201b:	48 83 fe 06          	cmp    $0x6,%rsi
    201f:	74 1d                	je     203e <phase_6+0xa7>
    2021:	8b 0c b4             	mov    (%rsp,%rsi,4),%ecx
    2024:	b8 01 00 00 00       	mov    $0x1,%eax
    2029:	48 8d 15 00 32 20 00 	lea    0x203200(%rip),%rdx        # 205230 <node1>
    2030:	83 f9 01             	cmp    $0x1,%ecx
    2033:	7f d2                	jg     2007 <phase_6+0x70>
    2035:	eb db                	jmp    2012 <phase_6+0x7b>
    2037:	be 00 00 00 00       	mov    $0x0,%esi
    203c:	eb e3                	jmp    2021 <phase_6+0x8a>
    203e:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
    2043:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    2048:	48 89 43 08          	mov    %rax,0x8(%rbx)
    204c:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
    2051:	48 89 50 08          	mov    %rdx,0x8(%rax)
    2055:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    205a:	48 89 42 08          	mov    %rax,0x8(%rdx)
    205e:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2063:	48 89 50 08          	mov    %rdx,0x8(%rax)
    2067:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    206c:	48 89 42 08          	mov    %rax,0x8(%rdx)
    2070:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
    2077:	00 
    2078:	bd 03 00 00 00       	mov    $0x3,%ebp
    207d:	eb 09                	jmp    2088 <phase_6+0xf1>
    207f:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
    2083:	83 ed 01             	sub    $0x1,%ebp
    2086:	74 11                	je     2099 <phase_6+0x102>
    2088:	48 8b 43 08          	mov    0x8(%rbx),%rax
    208c:	8b 00                	mov    (%rax),%eax
    208e:	39 03                	cmp    %eax,(%rbx)
    2090:	7d ed                	jge    207f <phase_6+0xe8>
    2092:	e8 11 05 00 00       	callq  25a8 <explode_bomb>
    2097:	eb e6                	jmp    207f <phase_6+0xe8>
    2099:	48 8b 43 08          	mov    0x8(%rbx),%rax
    209d:	8b 3b                	mov    (%rbx),%edi
    209f:	39 38                	cmp    %edi,(%rax)
    20a1:	7c 29                	jl     20cc <phase_6+0x135>
    20a3:	48 8b 43 08          	mov    0x8(%rbx),%rax
    20a7:	48 8b 50 08          	mov    0x8(%rax),%rdx
    20ab:	8b 3a                	mov    (%rdx),%edi
    20ad:	39 38                	cmp    %edi,(%rax)
    20af:	7f 22                	jg     20d3 <phase_6+0x13c>
    20b1:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    20b6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    20bd:	00 00 
    20bf:	75 19                	jne    20da <phase_6+0x143>
    20c1:	48 83 c4 68          	add    $0x68,%rsp
    20c5:	5b                   	pop    %rbx
    20c6:	5d                   	pop    %rbp
    20c7:	41 5c                	pop    %r12
    20c9:	41 5d                	pop    %r13
    20cb:	c3                   	retq   
    20cc:	e8 d7 04 00 00       	callq  25a8 <explode_bomb>
    20d1:	eb d0                	jmp    20a3 <phase_6+0x10c>
    20d3:	e8 d0 04 00 00       	callq  25a8 <explode_bomb>
    20d8:	eb d7                	jmp    20b1 <phase_6+0x11a>
    20da:	e8 d1 f7 ff ff       	callq  18b0 <__stack_chk_fail@plt>

00000000000020df <fun7>:
    20df:	48 85 ff             	test   %rdi,%rdi
    20e2:	74 34                	je     2118 <fun7+0x39>
    20e4:	48 83 ec 08          	sub    $0x8,%rsp
    20e8:	8b 17                	mov    (%rdi),%edx
    20ea:	39 f2                	cmp    %esi,%edx
    20ec:	7f 0e                	jg     20fc <fun7+0x1d>
    20ee:	b8 00 00 00 00       	mov    $0x0,%eax
    20f3:	39 f2                	cmp    %esi,%edx
    20f5:	75 12                	jne    2109 <fun7+0x2a>
    20f7:	48 83 c4 08          	add    $0x8,%rsp
    20fb:	c3                   	retq   
    20fc:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
    2100:	e8 da ff ff ff       	callq  20df <fun7>
    2105:	01 c0                	add    %eax,%eax
    2107:	eb ee                	jmp    20f7 <fun7+0x18>
    2109:	48 8b 7f 10          	mov    0x10(%rdi),%rdi
    210d:	e8 cd ff ff ff       	callq  20df <fun7>
    2112:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax
    2116:	eb df                	jmp    20f7 <fun7+0x18>
    2118:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    211d:	c3                   	retq   

000000000000211e <secret_phase>:
    211e:	53                   	push   %rbx
    211f:	e8 01 05 00 00       	callq  2625 <read_line>
    2124:	ba 0a 00 00 00       	mov    $0xa,%edx
    2129:	be 00 00 00 00       	mov    $0x0,%esi
    212e:	48 89 c7             	mov    %rax,%rdi
    2131:	e8 fa f7 ff ff       	callq  1930 <strtol@plt>
    2136:	48 89 c3             	mov    %rax,%rbx
    2139:	8d 40 ff             	lea    -0x1(%rax),%eax
    213c:	3d e8 03 00 00       	cmp    $0x3e8,%eax
    2141:	77 2b                	ja     216e <secret_phase+0x50>
    2143:	89 de                	mov    %ebx,%esi
    2145:	48 8d 3d 04 30 20 00 	lea    0x203004(%rip),%rdi        # 205150 <n1>
    214c:	e8 8e ff ff ff       	callq  20df <fun7>
    2151:	83 f8 06             	cmp    $0x6,%eax
    2154:	74 05                	je     215b <secret_phase+0x3d>
    2156:	e8 4d 04 00 00       	callq  25a8 <explode_bomb>
    215b:	48 8d 3d be 14 00 00 	lea    0x14be(%rip),%rdi        # 3620 <_IO_stdin_used+0x180>
    2162:	e8 29 f7 ff ff       	callq  1890 <puts@plt>
    2167:	e8 fd 05 00 00       	callq  2769 <phase_defused>
    216c:	5b                   	pop    %rbx
    216d:	c3                   	retq   
    216e:	e8 35 04 00 00       	callq  25a8 <explode_bomb>
    2173:	eb ce                	jmp    2143 <secret_phase+0x25>

0000000000002175 <sig_handler>:
    2175:	48 83 ec 08          	sub    $0x8,%rsp
    2179:	48 8d 3d 40 15 00 00 	lea    0x1540(%rip),%rdi        # 36c0 <array.3433+0x40>
    2180:	e8 0b f7 ff ff       	callq  1890 <puts@plt>
    2185:	bf 03 00 00 00       	mov    $0x3,%edi
    218a:	e8 31 f8 ff ff       	callq  19c0 <sleep@plt>
    218f:	48 8d 35 f3 16 00 00 	lea    0x16f3(%rip),%rsi        # 3889 <array.3433+0x209>
    2196:	bf 01 00 00 00       	mov    $0x1,%edi
    219b:	b8 00 00 00 00       	mov    $0x0,%eax
    21a0:	e8 bb f7 ff ff       	callq  1960 <__printf_chk@plt>
    21a5:	48 8b 3d d4 34 20 00 	mov    0x2034d4(%rip),%rdi        # 205680 <stdout@@GLIBC_2.2.5>
    21ac:	e8 8f f7 ff ff       	callq  1940 <fflush@plt>
    21b1:	bf 01 00 00 00       	mov    $0x1,%edi
    21b6:	e8 05 f8 ff ff       	callq  19c0 <sleep@plt>
    21bb:	48 8d 3d cf 16 00 00 	lea    0x16cf(%rip),%rdi        # 3891 <array.3433+0x211>
    21c2:	e8 c9 f6 ff ff       	callq  1890 <puts@plt>
    21c7:	bf 10 00 00 00       	mov    $0x10,%edi
    21cc:	e8 bf f7 ff ff       	callq  1990 <exit@plt>

00000000000021d1 <invalid_phase>:
    21d1:	48 83 ec 08          	sub    $0x8,%rsp
    21d5:	48 89 fa             	mov    %rdi,%rdx
    21d8:	48 8d 35 ba 16 00 00 	lea    0x16ba(%rip),%rsi        # 3899 <array.3433+0x219>
    21df:	bf 01 00 00 00       	mov    $0x1,%edi
    21e4:	b8 00 00 00 00       	mov    $0x0,%eax
    21e9:	e8 72 f7 ff ff       	callq  1960 <__printf_chk@plt>
    21ee:	bf 08 00 00 00       	mov    $0x8,%edi
    21f3:	e8 98 f7 ff ff       	callq  1990 <exit@plt>

00000000000021f8 <string_length>:
    21f8:	80 3f 00             	cmpb   $0x0,(%rdi)
    21fb:	74 12                	je     220f <string_length+0x17>
    21fd:	48 89 fa             	mov    %rdi,%rdx
    2200:	48 83 c2 01          	add    $0x1,%rdx
    2204:	89 d0                	mov    %edx,%eax
    2206:	29 f8                	sub    %edi,%eax
    2208:	80 3a 00             	cmpb   $0x0,(%rdx)
    220b:	75 f3                	jne    2200 <string_length+0x8>
    220d:	f3 c3                	repz retq 
    220f:	b8 00 00 00 00       	mov    $0x0,%eax
    2214:	c3                   	retq   

0000000000002215 <check_division_equal>:
    2215:	89 f8                	mov    %edi,%eax
    2217:	89 d1                	mov    %edx,%ecx
    2219:	99                   	cltd   
    221a:	f7 fe                	idiv   %esi
    221c:	39 c8                	cmp    %ecx,%eax
    221e:	0f 94 c0             	sete   %al
    2221:	0f b6 c0             	movzbl %al,%eax
    2224:	c3                   	retq   

0000000000002225 <check_multiplication_equal>:
    2225:	0f af fe             	imul   %esi,%edi
    2228:	39 d7                	cmp    %edx,%edi
    222a:	0f 94 c0             	sete   %al
    222d:	0f b6 c0             	movzbl %al,%eax
    2230:	c3                   	retq   

0000000000002231 <check_substraction_equal>:
    2231:	29 f7                	sub    %esi,%edi
    2233:	39 d7                	cmp    %edx,%edi
    2235:	0f 94 c0             	sete   %al
    2238:	0f b6 c0             	movzbl %al,%eax
    223b:	c3                   	retq   

000000000000223c <reverse_string>:
    223c:	48 89 fe             	mov    %rdi,%rsi
    223f:	48 85 ff             	test   %rdi,%rdi
    2242:	74 40                	je     2284 <reverse_string+0x48>
    2244:	80 3f 00             	cmpb   $0x0,(%rdi)
    2247:	74 3b                	je     2284 <reverse_string+0x48>
    2249:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
    2250:	b8 00 00 00 00       	mov    $0x0,%eax
    2255:	f2 ae                	repnz scas %es:(%rdi),%al
    2257:	48 89 ca             	mov    %rcx,%rdx
    225a:	48 f7 d2             	not    %rdx
    225d:	48 8d 4c 16 fe       	lea    -0x2(%rsi,%rdx,1),%rcx
    2262:	48 39 ce             	cmp    %rcx,%rsi
    2265:	73 1d                	jae    2284 <reverse_string+0x48>
    2267:	48 89 f2             	mov    %rsi,%rdx
    226a:	0f b6 02             	movzbl (%rdx),%eax
    226d:	32 01                	xor    (%rcx),%al
    226f:	88 02                	mov    %al,(%rdx)
    2271:	32 01                	xor    (%rcx),%al
    2273:	88 01                	mov    %al,(%rcx)
    2275:	30 02                	xor    %al,(%rdx)
    2277:	48 83 c2 01          	add    $0x1,%rdx
    227b:	48 83 e9 01          	sub    $0x1,%rcx
    227f:	48 39 ca             	cmp    %rcx,%rdx
    2282:	72 e6                	jb     226a <reverse_string+0x2e>
    2284:	48 89 f0             	mov    %rsi,%rax
    2287:	c3                   	retq   

0000000000002288 <strings_not_equal>:
    2288:	41 54                	push   %r12
    228a:	55                   	push   %rbp
    228b:	53                   	push   %rbx
    228c:	48 89 fb             	mov    %rdi,%rbx
    228f:	48 89 f5             	mov    %rsi,%rbp
    2292:	e8 61 ff ff ff       	callq  21f8 <string_length>
    2297:	41 89 c4             	mov    %eax,%r12d
    229a:	48 89 ef             	mov    %rbp,%rdi
    229d:	e8 56 ff ff ff       	callq  21f8 <string_length>
    22a2:	ba 01 00 00 00       	mov    $0x1,%edx
    22a7:	41 39 c4             	cmp    %eax,%r12d
    22aa:	74 07                	je     22b3 <strings_not_equal+0x2b>
    22ac:	89 d0                	mov    %edx,%eax
    22ae:	5b                   	pop    %rbx
    22af:	5d                   	pop    %rbp
    22b0:	41 5c                	pop    %r12
    22b2:	c3                   	retq   
    22b3:	0f b6 03             	movzbl (%rbx),%eax
    22b6:	84 c0                	test   %al,%al
    22b8:	74 27                	je     22e1 <strings_not_equal+0x59>
    22ba:	3a 45 00             	cmp    0x0(%rbp),%al
    22bd:	75 29                	jne    22e8 <strings_not_equal+0x60>
    22bf:	48 83 c3 01          	add    $0x1,%rbx
    22c3:	48 83 c5 01          	add    $0x1,%rbp
    22c7:	0f b6 03             	movzbl (%rbx),%eax
    22ca:	84 c0                	test   %al,%al
    22cc:	74 0c                	je     22da <strings_not_equal+0x52>
    22ce:	38 45 00             	cmp    %al,0x0(%rbp)
    22d1:	74 ec                	je     22bf <strings_not_equal+0x37>
    22d3:	ba 01 00 00 00       	mov    $0x1,%edx
    22d8:	eb d2                	jmp    22ac <strings_not_equal+0x24>
    22da:	ba 00 00 00 00       	mov    $0x0,%edx
    22df:	eb cb                	jmp    22ac <strings_not_equal+0x24>
    22e1:	ba 00 00 00 00       	mov    $0x0,%edx
    22e6:	eb c4                	jmp    22ac <strings_not_equal+0x24>
    22e8:	ba 01 00 00 00       	mov    $0x1,%edx
    22ed:	eb bd                	jmp    22ac <strings_not_equal+0x24>

00000000000022ef <from_char_to_int>:
    22ef:	40 0f be c7          	movsbl %dil,%eax
    22f3:	40 80 ff 69          	cmp    $0x69,%dil
    22f7:	7f 0e                	jg     2307 <from_char_to_int+0x18>
    22f9:	83 ef 33             	sub    $0x33,%edi
    22fc:	8d 50 fd             	lea    -0x3(%rax),%edx
    22ff:	40 80 ff 0a          	cmp    $0xa,%dil
    2303:	0f 42 c2             	cmovb  %edx,%eax
    2306:	c3                   	retq   
    2307:	83 e8 09             	sub    $0x9,%eax
    230a:	c3                   	retq   

000000000000230b <initialize_bomb>:
    230b:	55                   	push   %rbp
    230c:	53                   	push   %rbx
    230d:	48 81 ec 58 20 00 00 	sub    $0x2058,%rsp
    2314:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    231b:	00 00 
    231d:	48 89 84 24 48 20 00 	mov    %rax,0x2048(%rsp)
    2324:	00 
    2325:	31 c0                	xor    %eax,%eax
    2327:	48 8d 35 47 fe ff ff 	lea    -0x1b9(%rip),%rsi        # 2175 <sig_handler>
    232e:	bf 02 00 00 00       	mov    $0x2,%edi
    2333:	e8 c8 f5 ff ff       	callq  1900 <signal@plt>
    2338:	48 89 e7             	mov    %rsp,%rdi
    233b:	be 40 00 00 00       	mov    $0x40,%esi
    2340:	e8 3b f6 ff ff       	callq  1980 <gethostname@plt>
    2345:	85 c0                	test   %eax,%eax
    2347:	75 45                	jne    238e <initialize_bomb+0x83>
    2349:	48 8b 3d 30 2f 20 00 	mov    0x202f30(%rip),%rdi        # 205280 <host_table>
    2350:	48 8d 1d 31 2f 20 00 	lea    0x202f31(%rip),%rbx        # 205288 <host_table+0x8>
    2357:	48 89 e5             	mov    %rsp,%rbp
    235a:	48 85 ff             	test   %rdi,%rdi
    235d:	74 19                	je     2378 <initialize_bomb+0x6d>
    235f:	48 89 ee             	mov    %rbp,%rsi
    2362:	e8 f9 f4 ff ff       	callq  1860 <strcasecmp@plt>
    2367:	85 c0                	test   %eax,%eax
    2369:	74 5e                	je     23c9 <initialize_bomb+0xbe>
    236b:	48 83 c3 08          	add    $0x8,%rbx
    236f:	48 8b 7b f8          	mov    -0x8(%rbx),%rdi
    2373:	48 85 ff             	test   %rdi,%rdi
    2376:	75 e7                	jne    235f <initialize_bomb+0x54>
    2378:	48 8d 3d b1 13 00 00 	lea    0x13b1(%rip),%rdi        # 3730 <array.3433+0xb0>
    237f:	e8 0c f5 ff ff       	callq  1890 <puts@plt>
    2384:	bf 08 00 00 00       	mov    $0x8,%edi
    2389:	e8 02 f6 ff ff       	callq  1990 <exit@plt>
    238e:	48 8d 3d 63 13 00 00 	lea    0x1363(%rip),%rdi        # 36f8 <array.3433+0x78>
    2395:	e8 f6 f4 ff ff       	callq  1890 <puts@plt>
    239a:	bf 08 00 00 00       	mov    $0x8,%edi
    239f:	e8 ec f5 ff ff       	callq  1990 <exit@plt>
    23a4:	48 8d 54 24 40       	lea    0x40(%rsp),%rdx
    23a9:	48 8d 35 fa 14 00 00 	lea    0x14fa(%rip),%rsi        # 38aa <array.3433+0x22a>
    23b0:	bf 01 00 00 00       	mov    $0x1,%edi
    23b5:	b8 00 00 00 00       	mov    $0x0,%eax
    23ba:	e8 a1 f5 ff ff       	callq  1960 <__printf_chk@plt>
    23bf:	bf 08 00 00 00       	mov    $0x8,%edi
    23c4:	e8 c7 f5 ff ff       	callq  1990 <exit@plt>
    23c9:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    23ce:	e8 b2 0d 00 00       	callq  3185 <init_driver>
    23d3:	85 c0                	test   %eax,%eax
    23d5:	78 cd                	js     23a4 <initialize_bomb+0x99>
    23d7:	48 8b 84 24 48 20 00 	mov    0x2048(%rsp),%rax
    23de:	00 
    23df:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    23e6:	00 00 
    23e8:	75 0a                	jne    23f4 <initialize_bomb+0xe9>
    23ea:	48 81 c4 58 20 00 00 	add    $0x2058,%rsp
    23f1:	5b                   	pop    %rbx
    23f2:	5d                   	pop    %rbp
    23f3:	c3                   	retq   
    23f4:	e8 b7 f4 ff ff       	callq  18b0 <__stack_chk_fail@plt>

00000000000023f9 <initialize_bomb_solve>:
    23f9:	f3 c3                	repz retq 

00000000000023fb <blank_line>:
    23fb:	55                   	push   %rbp
    23fc:	53                   	push   %rbx
    23fd:	48 83 ec 08          	sub    $0x8,%rsp
    2401:	48 89 fd             	mov    %rdi,%rbp
    2404:	0f b6 5d 00          	movzbl 0x0(%rbp),%ebx
    2408:	84 db                	test   %bl,%bl
    240a:	74 1e                	je     242a <blank_line+0x2f>
    240c:	e8 bf f5 ff ff       	callq  19d0 <__ctype_b_loc@plt>
    2411:	48 83 c5 01          	add    $0x1,%rbp
    2415:	48 0f be db          	movsbq %bl,%rbx
    2419:	48 8b 00             	mov    (%rax),%rax
    241c:	f6 44 58 01 20       	testb  $0x20,0x1(%rax,%rbx,2)
    2421:	75 e1                	jne    2404 <blank_line+0x9>
    2423:	b8 00 00 00 00       	mov    $0x0,%eax
    2428:	eb 05                	jmp    242f <blank_line+0x34>
    242a:	b8 01 00 00 00       	mov    $0x1,%eax
    242f:	48 83 c4 08          	add    $0x8,%rsp
    2433:	5b                   	pop    %rbx
    2434:	5d                   	pop    %rbp
    2435:	c3                   	retq   

0000000000002436 <skip>:
    2436:	55                   	push   %rbp
    2437:	53                   	push   %rbx
    2438:	48 83 ec 08          	sub    $0x8,%rsp
    243c:	48 8d 2d 7d 32 20 00 	lea    0x20327d(%rip),%rbp        # 2056c0 <input_strings>
    2443:	48 63 05 62 32 20 00 	movslq 0x203262(%rip),%rax        # 2056ac <num_input_strings>
    244a:	48 8d 3c 80          	lea    (%rax,%rax,4),%rdi
    244e:	48 c1 e7 04          	shl    $0x4,%rdi
    2452:	48 01 ef             	add    %rbp,%rdi
    2455:	48 8b 15 54 32 20 00 	mov    0x203254(%rip),%rdx        # 2056b0 <infile>
    245c:	be 50 00 00 00       	mov    $0x50,%esi
    2461:	e8 8a f4 ff ff       	callq  18f0 <fgets@plt>
    2466:	48 89 c3             	mov    %rax,%rbx
    2469:	48 85 c0             	test   %rax,%rax
    246c:	74 0c                	je     247a <skip+0x44>
    246e:	48 89 c7             	mov    %rax,%rdi
    2471:	e8 85 ff ff ff       	callq  23fb <blank_line>
    2476:	85 c0                	test   %eax,%eax
    2478:	75 c9                	jne    2443 <skip+0xd>
    247a:	48 89 d8             	mov    %rbx,%rax
    247d:	48 83 c4 08          	add    $0x8,%rsp
    2481:	5b                   	pop    %rbx
    2482:	5d                   	pop    %rbp
    2483:	c3                   	retq   

0000000000002484 <send_msg>:
    2484:	53                   	push   %rbx
    2485:	48 81 ec 10 40 00 00 	sub    $0x4010,%rsp
    248c:	41 89 f8             	mov    %edi,%r8d
    248f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2496:	00 00 
    2498:	48 89 84 24 08 40 00 	mov    %rax,0x4008(%rsp)
    249f:	00 
    24a0:	31 c0                	xor    %eax,%eax
    24a2:	8b 35 04 32 20 00    	mov    0x203204(%rip),%esi        # 2056ac <num_input_strings>
    24a8:	8d 46 ff             	lea    -0x1(%rsi),%eax
    24ab:	48 98                	cltq   
    24ad:	48 8d 14 80          	lea    (%rax,%rax,4),%rdx
    24b1:	48 c1 e2 04          	shl    $0x4,%rdx
    24b5:	48 8d 05 04 32 20 00 	lea    0x203204(%rip),%rax        # 2056c0 <input_strings>
    24bc:	48 01 c2             	add    %rax,%rdx
    24bf:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
    24c6:	b8 00 00 00 00       	mov    $0x0,%eax
    24cb:	48 89 d7             	mov    %rdx,%rdi
    24ce:	f2 ae                	repnz scas %es:(%rdi),%al
    24d0:	48 89 c8             	mov    %rcx,%rax
    24d3:	48 f7 d0             	not    %rax
    24d6:	48 83 c0 63          	add    $0x63,%rax
    24da:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
    24e0:	0f 87 86 00 00 00    	ja     256c <send_msg+0xe8>
    24e6:	45 85 c0             	test   %r8d,%r8d
    24e9:	4c 8d 0d d4 13 00 00 	lea    0x13d4(%rip),%r9        # 38c4 <array.3433+0x244>
    24f0:	48 8d 05 d5 13 00 00 	lea    0x13d5(%rip),%rax        # 38cc <array.3433+0x24c>
    24f7:	4c 0f 44 c8          	cmove  %rax,%r9
    24fb:	48 89 e3             	mov    %rsp,%rbx
    24fe:	52                   	push   %rdx
    24ff:	56                   	push   %rsi
    2500:	44 8b 05 39 2c 20 00 	mov    0x202c39(%rip),%r8d        # 205140 <bomb_id>
    2507:	48 8d 0d c7 13 00 00 	lea    0x13c7(%rip),%rcx        # 38d5 <array.3433+0x255>
    250e:	ba 00 20 00 00       	mov    $0x2000,%edx
    2513:	be 01 00 00 00       	mov    $0x1,%esi
    2518:	48 89 df             	mov    %rbx,%rdi
    251b:	b8 00 00 00 00       	mov    $0x0,%eax
    2520:	e8 bb f4 ff ff       	callq  19e0 <__sprintf_chk@plt>
    2525:	4c 8d 84 24 10 20 00 	lea    0x2010(%rsp),%r8
    252c:	00 
    252d:	b9 00 00 00 00       	mov    $0x0,%ecx
    2532:	48 89 da             	mov    %rbx,%rdx
    2535:	48 8d 35 e4 2b 20 00 	lea    0x202be4(%rip),%rsi        # 205120 <user_password>
    253c:	48 8d 3d f5 2b 20 00 	lea    0x202bf5(%rip),%rdi        # 205138 <userid>
    2543:	e8 46 0e 00 00       	callq  338e <driver_post>
    2548:	48 83 c4 10          	add    $0x10,%rsp
    254c:	85 c0                	test   %eax,%eax
    254e:	78 3c                	js     258c <send_msg+0x108>
    2550:	48 8b 84 24 08 40 00 	mov    0x4008(%rsp),%rax
    2557:	00 
    2558:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    255f:	00 00 
    2561:	75 40                	jne    25a3 <send_msg+0x11f>
    2563:	48 81 c4 10 40 00 00 	add    $0x4010,%rsp
    256a:	5b                   	pop    %rbx
    256b:	c3                   	retq   
    256c:	48 8d 35 f5 11 00 00 	lea    0x11f5(%rip),%rsi        # 3768 <array.3433+0xe8>
    2573:	bf 01 00 00 00       	mov    $0x1,%edi
    2578:	b8 00 00 00 00       	mov    $0x0,%eax
    257d:	e8 de f3 ff ff       	callq  1960 <__printf_chk@plt>
    2582:	bf 08 00 00 00       	mov    $0x8,%edi
    2587:	e8 04 f4 ff ff       	callq  1990 <exit@plt>
    258c:	48 8d bc 24 00 20 00 	lea    0x2000(%rsp),%rdi
    2593:	00 
    2594:	e8 f7 f2 ff ff       	callq  1890 <puts@plt>
    2599:	bf 00 00 00 00       	mov    $0x0,%edi
    259e:	e8 ed f3 ff ff       	callq  1990 <exit@plt>
    25a3:	e8 08 f3 ff ff       	callq  18b0 <__stack_chk_fail@plt>

00000000000025a8 <explode_bomb>:
    25a8:	48 83 ec 08          	sub    $0x8,%rsp
    25ac:	48 8d 3d 2e 13 00 00 	lea    0x132e(%rip),%rdi        # 38e1 <array.3433+0x261>
    25b3:	e8 d8 f2 ff ff       	callq  1890 <puts@plt>
    25b8:	48 8d 3d 2b 13 00 00 	lea    0x132b(%rip),%rdi        # 38ea <array.3433+0x26a>
    25bf:	e8 cc f2 ff ff       	callq  1890 <puts@plt>
    25c4:	bf 00 00 00 00       	mov    $0x0,%edi
    25c9:	e8 b6 fe ff ff       	callq  2484 <send_msg>
    25ce:	48 8d 3d bb 11 00 00 	lea    0x11bb(%rip),%rdi        # 3790 <array.3433+0x110>
    25d5:	e8 b6 f2 ff ff       	callq  1890 <puts@plt>
    25da:	bf 08 00 00 00       	mov    $0x8,%edi
    25df:	e8 ac f3 ff ff       	callq  1990 <exit@plt>

00000000000025e4 <read_six_numbers>:
    25e4:	48 83 ec 08          	sub    $0x8,%rsp
    25e8:	48 89 f2             	mov    %rsi,%rdx
    25eb:	48 8d 4e 04          	lea    0x4(%rsi),%rcx
    25ef:	48 8d 46 14          	lea    0x14(%rsi),%rax
    25f3:	50                   	push   %rax
    25f4:	48 8d 46 10          	lea    0x10(%rsi),%rax
    25f8:	50                   	push   %rax
    25f9:	4c 8d 4e 0c          	lea    0xc(%rsi),%r9
    25fd:	4c 8d 46 08          	lea    0x8(%rsi),%r8
    2601:	48 8d 35 f9 12 00 00 	lea    0x12f9(%rip),%rsi        # 3901 <array.3433+0x281>
    2608:	b8 00 00 00 00       	mov    $0x0,%eax
    260d:	e8 3e f3 ff ff       	callq  1950 <__isoc99_sscanf@plt>
    2612:	48 83 c4 10          	add    $0x10,%rsp
    2616:	83 f8 05             	cmp    $0x5,%eax
    2619:	7e 05                	jle    2620 <read_six_numbers+0x3c>
    261b:	48 83 c4 08          	add    $0x8,%rsp
    261f:	c3                   	retq   
    2620:	e8 83 ff ff ff       	callq  25a8 <explode_bomb>

0000000000002625 <read_line>:
    2625:	48 83 ec 08          	sub    $0x8,%rsp
    2629:	b8 00 00 00 00       	mov    $0x0,%eax
    262e:	e8 03 fe ff ff       	callq  2436 <skip>
    2633:	48 85 c0             	test   %rax,%rax
    2636:	74 6f                	je     26a7 <read_line+0x82>
    2638:	8b 35 6e 30 20 00    	mov    0x20306e(%rip),%esi        # 2056ac <num_input_strings>
    263e:	48 63 c6             	movslq %esi,%rax
    2641:	48 8d 14 80          	lea    (%rax,%rax,4),%rdx
    2645:	48 c1 e2 04          	shl    $0x4,%rdx
    2649:	48 8d 05 70 30 20 00 	lea    0x203070(%rip),%rax        # 2056c0 <input_strings>
    2650:	48 01 c2             	add    %rax,%rdx
    2653:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
    265a:	b8 00 00 00 00       	mov    $0x0,%eax
    265f:	48 89 d7             	mov    %rdx,%rdi
    2662:	f2 ae                	repnz scas %es:(%rdi),%al
    2664:	48 f7 d1             	not    %rcx
    2667:	48 83 e9 01          	sub    $0x1,%rcx
    266b:	83 f9 4e             	cmp    $0x4e,%ecx
    266e:	0f 8f ab 00 00 00    	jg     271f <read_line+0xfa>
    2674:	83 e9 01             	sub    $0x1,%ecx
    2677:	48 63 c9             	movslq %ecx,%rcx
    267a:	48 63 c6             	movslq %esi,%rax
    267d:	48 8d 04 80          	lea    (%rax,%rax,4),%rax
    2681:	48 c1 e0 04          	shl    $0x4,%rax
    2685:	48 89 c7             	mov    %rax,%rdi
    2688:	48 8d 05 31 30 20 00 	lea    0x203031(%rip),%rax        # 2056c0 <input_strings>
    268f:	48 01 f8             	add    %rdi,%rax
    2692:	c6 04 08 00          	movb   $0x0,(%rax,%rcx,1)
    2696:	83 c6 01             	add    $0x1,%esi
    2699:	89 35 0d 30 20 00    	mov    %esi,0x20300d(%rip)        # 2056ac <num_input_strings>
    269f:	48 89 d0             	mov    %rdx,%rax
    26a2:	48 83 c4 08          	add    $0x8,%rsp
    26a6:	c3                   	retq   
    26a7:	48 8b 05 e2 2f 20 00 	mov    0x202fe2(%rip),%rax        # 205690 <stdin@@GLIBC_2.2.5>
    26ae:	48 39 05 fb 2f 20 00 	cmp    %rax,0x202ffb(%rip)        # 2056b0 <infile>
    26b5:	74 1b                	je     26d2 <read_line+0xad>
    26b7:	48 8d 3d 73 12 00 00 	lea    0x1273(%rip),%rdi        # 3931 <array.3433+0x2b1>
    26be:	e8 8d f1 ff ff       	callq  1850 <getenv@plt>
    26c3:	48 85 c0             	test   %rax,%rax
    26c6:	74 20                	je     26e8 <read_line+0xc3>
    26c8:	bf 00 00 00 00       	mov    $0x0,%edi
    26cd:	e8 be f2 ff ff       	callq  1990 <exit@plt>
    26d2:	48 8d 3d 3a 12 00 00 	lea    0x123a(%rip),%rdi        # 3913 <array.3433+0x293>
    26d9:	e8 b2 f1 ff ff       	callq  1890 <puts@plt>
    26de:	bf 08 00 00 00       	mov    $0x8,%edi
    26e3:	e8 a8 f2 ff ff       	callq  1990 <exit@plt>
    26e8:	48 8b 05 a1 2f 20 00 	mov    0x202fa1(%rip),%rax        # 205690 <stdin@@GLIBC_2.2.5>
    26ef:	48 89 05 ba 2f 20 00 	mov    %rax,0x202fba(%rip)        # 2056b0 <infile>
    26f6:	b8 00 00 00 00       	mov    $0x0,%eax
    26fb:	e8 36 fd ff ff       	callq  2436 <skip>
    2700:	48 85 c0             	test   %rax,%rax
    2703:	0f 85 2f ff ff ff    	jne    2638 <read_line+0x13>
    2709:	48 8d 3d 03 12 00 00 	lea    0x1203(%rip),%rdi        # 3913 <array.3433+0x293>
    2710:	e8 7b f1 ff ff       	callq  1890 <puts@plt>
    2715:	bf 00 00 00 00       	mov    $0x0,%edi
    271a:	e8 71 f2 ff ff       	callq  1990 <exit@plt>
    271f:	48 8d 3d 16 12 00 00 	lea    0x1216(%rip),%rdi        # 393c <array.3433+0x2bc>
    2726:	e8 65 f1 ff ff       	callq  1890 <puts@plt>
    272b:	8b 05 7b 2f 20 00    	mov    0x202f7b(%rip),%eax        # 2056ac <num_input_strings>
    2731:	8d 50 01             	lea    0x1(%rax),%edx
    2734:	89 15 72 2f 20 00    	mov    %edx,0x202f72(%rip)        # 2056ac <num_input_strings>
    273a:	48 98                	cltq   
    273c:	48 6b c0 50          	imul   $0x50,%rax,%rax
    2740:	48 8d 15 79 2f 20 00 	lea    0x202f79(%rip),%rdx        # 2056c0 <input_strings>
    2747:	48 be 2a 2a 2a 74 72 	movabs $0x636e7572742a2a2a,%rsi
    274e:	75 6e 63 
    2751:	48 bf 61 74 65 64 2a 	movabs $0x2a2a2a64657461,%rdi
    2758:	2a 2a 00 
    275b:	48 89 34 02          	mov    %rsi,(%rdx,%rax,1)
    275f:	48 89 7c 02 08       	mov    %rdi,0x8(%rdx,%rax,1)
    2764:	e8 3f fe ff ff       	callq  25a8 <explode_bomb>

0000000000002769 <phase_defused>:
    2769:	48 83 ec 78          	sub    $0x78,%rsp
    276d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2774:	00 00 
    2776:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    277b:	31 c0                	xor    %eax,%eax
    277d:	bf 01 00 00 00       	mov    $0x1,%edi
    2782:	e8 fd fc ff ff       	callq  2484 <send_msg>
    2787:	83 3d 1e 2f 20 00 06 	cmpl   $0x6,0x202f1e(%rip)        # 2056ac <num_input_strings>
    278e:	74 19                	je     27a9 <phase_defused+0x40>
    2790:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
    2795:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    279c:	00 00 
    279e:	0f 85 84 00 00 00    	jne    2828 <phase_defused+0xbf>
    27a4:	48 83 c4 78          	add    $0x78,%rsp
    27a8:	c3                   	retq   
    27a9:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
    27ae:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
    27b3:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
    27b8:	48 8d 35 98 11 00 00 	lea    0x1198(%rip),%rsi        # 3957 <array.3433+0x2d7>
    27bf:	48 8d 3d ea 2f 20 00 	lea    0x202fea(%rip),%rdi        # 2057b0 <input_strings+0xf0>
    27c6:	b8 00 00 00 00       	mov    $0x0,%eax
    27cb:	e8 80 f1 ff ff       	callq  1950 <__isoc99_sscanf@plt>
    27d0:	83 f8 03             	cmp    $0x3,%eax
    27d3:	74 1a                	je     27ef <phase_defused+0x86>
    27d5:	48 8d 3d 3c 10 00 00 	lea    0x103c(%rip),%rdi        # 3818 <array.3433+0x198>
    27dc:	e8 af f0 ff ff       	callq  1890 <puts@plt>
    27e1:	48 8d 3d 60 10 00 00 	lea    0x1060(%rip),%rdi        # 3848 <array.3433+0x1c8>
    27e8:	e8 a3 f0 ff ff       	callq  1890 <puts@plt>
    27ed:	eb a1                	jmp    2790 <phase_defused+0x27>
    27ef:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
    27f4:	48 8d 35 65 11 00 00 	lea    0x1165(%rip),%rsi        # 3960 <array.3433+0x2e0>
    27fb:	e8 88 fa ff ff       	callq  2288 <strings_not_equal>
    2800:	85 c0                	test   %eax,%eax
    2802:	75 d1                	jne    27d5 <phase_defused+0x6c>
    2804:	48 8d 3d ad 0f 00 00 	lea    0xfad(%rip),%rdi        # 37b8 <array.3433+0x138>
    280b:	e8 80 f0 ff ff       	callq  1890 <puts@plt>
    2810:	48 8d 3d c9 0f 00 00 	lea    0xfc9(%rip),%rdi        # 37e0 <array.3433+0x160>
    2817:	e8 74 f0 ff ff       	callq  1890 <puts@plt>
    281c:	b8 00 00 00 00       	mov    $0x0,%eax
    2821:	e8 f8 f8 ff ff       	callq  211e <secret_phase>
    2826:	eb ad                	jmp    27d5 <phase_defused+0x6c>
    2828:	e8 83 f0 ff ff       	callq  18b0 <__stack_chk_fail@plt>

000000000000282d <sigalrm_handler>:
    282d:	48 83 ec 08          	sub    $0x8,%rsp
    2831:	b9 00 00 00 00       	mov    $0x0,%ecx
    2836:	48 8d 15 7b 14 00 00 	lea    0x147b(%rip),%rdx        # 3cb8 <array.3433+0x638>
    283d:	be 01 00 00 00       	mov    $0x1,%esi
    2842:	48 8b 3d 57 2e 20 00 	mov    0x202e57(%rip),%rdi        # 2056a0 <stderr@@GLIBC_2.2.5>
    2849:	b8 00 00 00 00       	mov    $0x0,%eax
    284e:	e8 5d f1 ff ff       	callq  19b0 <__fprintf_chk@plt>
    2853:	bf 01 00 00 00       	mov    $0x1,%edi
    2858:	e8 33 f1 ff ff       	callq  1990 <exit@plt>

000000000000285d <rio_readlineb>:
    285d:	41 56                	push   %r14
    285f:	41 55                	push   %r13
    2861:	41 54                	push   %r12
    2863:	55                   	push   %rbp
    2864:	53                   	push   %rbx
    2865:	48 89 fb             	mov    %rdi,%rbx
    2868:	49 89 f4             	mov    %rsi,%r12
    286b:	49 89 d6             	mov    %rdx,%r14
    286e:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    2874:	48 8d 6f 10          	lea    0x10(%rdi),%rbp
    2878:	48 83 fa 01          	cmp    $0x1,%rdx
    287c:	77 0c                	ja     288a <rio_readlineb+0x2d>
    287e:	eb 60                	jmp    28e0 <rio_readlineb+0x83>
    2880:	e8 eb ef ff ff       	callq  1870 <__errno_location@plt>
    2885:	83 38 04             	cmpl   $0x4,(%rax)
    2888:	75 67                	jne    28f1 <rio_readlineb+0x94>
    288a:	8b 43 04             	mov    0x4(%rbx),%eax
    288d:	85 c0                	test   %eax,%eax
    288f:	7f 20                	jg     28b1 <rio_readlineb+0x54>
    2891:	ba 00 20 00 00       	mov    $0x2000,%edx
    2896:	48 89 ee             	mov    %rbp,%rsi
    2899:	8b 3b                	mov    (%rbx),%edi
    289b:	e8 40 f0 ff ff       	callq  18e0 <read@plt>
    28a0:	89 43 04             	mov    %eax,0x4(%rbx)
    28a3:	85 c0                	test   %eax,%eax
    28a5:	78 d9                	js     2880 <rio_readlineb+0x23>
    28a7:	85 c0                	test   %eax,%eax
    28a9:	74 4f                	je     28fa <rio_readlineb+0x9d>
    28ab:	48 89 6b 08          	mov    %rbp,0x8(%rbx)
    28af:	eb d9                	jmp    288a <rio_readlineb+0x2d>
    28b1:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    28b5:	0f b6 0a             	movzbl (%rdx),%ecx
    28b8:	48 83 c2 01          	add    $0x1,%rdx
    28bc:	48 89 53 08          	mov    %rdx,0x8(%rbx)
    28c0:	83 e8 01             	sub    $0x1,%eax
    28c3:	89 43 04             	mov    %eax,0x4(%rbx)
    28c6:	49 83 c4 01          	add    $0x1,%r12
    28ca:	41 88 4c 24 ff       	mov    %cl,-0x1(%r12)
    28cf:	80 f9 0a             	cmp    $0xa,%cl
    28d2:	74 0c                	je     28e0 <rio_readlineb+0x83>
    28d4:	41 83 c5 01          	add    $0x1,%r13d
    28d8:	49 63 c5             	movslq %r13d,%rax
    28db:	4c 39 f0             	cmp    %r14,%rax
    28de:	72 aa                	jb     288a <rio_readlineb+0x2d>
    28e0:	41 c6 04 24 00       	movb   $0x0,(%r12)
    28e5:	49 63 c5             	movslq %r13d,%rax
    28e8:	5b                   	pop    %rbx
    28e9:	5d                   	pop    %rbp
    28ea:	41 5c                	pop    %r12
    28ec:	41 5d                	pop    %r13
    28ee:	41 5e                	pop    %r14
    28f0:	c3                   	retq   
    28f1:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
    28f8:	eb 05                	jmp    28ff <rio_readlineb+0xa2>
    28fa:	b8 00 00 00 00       	mov    $0x0,%eax
    28ff:	85 c0                	test   %eax,%eax
    2901:	75 0d                	jne    2910 <rio_readlineb+0xb3>
    2903:	b8 00 00 00 00       	mov    $0x0,%eax
    2908:	41 83 fd 01          	cmp    $0x1,%r13d
    290c:	75 d2                	jne    28e0 <rio_readlineb+0x83>
    290e:	eb d8                	jmp    28e8 <rio_readlineb+0x8b>
    2910:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
    2917:	eb cf                	jmp    28e8 <rio_readlineb+0x8b>

0000000000002919 <submitr>:
    2919:	41 57                	push   %r15
    291b:	41 56                	push   %r14
    291d:	41 55                	push   %r13
    291f:	41 54                	push   %r12
    2921:	55                   	push   %rbp
    2922:	53                   	push   %rbx
    2923:	48 81 ec 78 a0 00 00 	sub    $0xa078,%rsp
    292a:	49 89 fd             	mov    %rdi,%r13
    292d:	89 f5                	mov    %esi,%ebp
    292f:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    2934:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    2939:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    293e:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
    2943:	48 8b 9c 24 b0 a0 00 	mov    0xa0b0(%rsp),%rbx
    294a:	00 
    294b:	4c 8b bc 24 b8 a0 00 	mov    0xa0b8(%rsp),%r15
    2952:	00 
    2953:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    295a:	00 00 
    295c:	48 89 84 24 68 a0 00 	mov    %rax,0xa068(%rsp)
    2963:	00 
    2964:	31 c0                	xor    %eax,%eax
    2966:	c7 44 24 3c 00 00 00 	movl   $0x0,0x3c(%rsp)
    296d:	00 
    296e:	ba 00 00 00 00       	mov    $0x0,%edx
    2973:	be 01 00 00 00       	mov    $0x1,%esi
    2978:	bf 02 00 00 00       	mov    $0x2,%edi
    297d:	e8 6e f0 ff ff       	callq  19f0 <socket@plt>
    2982:	85 c0                	test   %eax,%eax
    2984:	0f 88 35 01 00 00    	js     2abf <submitr+0x1a6>
    298a:	41 89 c4             	mov    %eax,%r12d
    298d:	4c 89 ef             	mov    %r13,%rdi
    2990:	e8 7b ef ff ff       	callq  1910 <gethostbyname@plt>
    2995:	48 85 c0             	test   %rax,%rax
    2998:	0f 84 71 01 00 00    	je     2b0f <submitr+0x1f6>
    299e:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
    29a3:	48 c7 44 24 42 00 00 	movq   $0x0,0x42(%rsp)
    29aa:	00 00 
    29ac:	c7 44 24 4a 00 00 00 	movl   $0x0,0x4a(%rsp)
    29b3:	00 
    29b4:	66 c7 44 24 4e 00 00 	movw   $0x0,0x4e(%rsp)
    29bb:	66 c7 44 24 40 02 00 	movw   $0x2,0x40(%rsp)
    29c2:	48 63 50 14          	movslq 0x14(%rax),%rdx
    29c6:	48 8b 40 18          	mov    0x18(%rax),%rax
    29ca:	48 8d 7c 24 44       	lea    0x44(%rsp),%rdi
    29cf:	b9 0c 00 00 00       	mov    $0xc,%ecx
    29d4:	48 8b 30             	mov    (%rax),%rsi
    29d7:	e8 44 ef ff ff       	callq  1920 <__memmove_chk@plt>
    29dc:	66 c1 cd 08          	ror    $0x8,%bp
    29e0:	66 89 6c 24 42       	mov    %bp,0x42(%rsp)
    29e5:	ba 10 00 00 00       	mov    $0x10,%edx
    29ea:	4c 89 ee             	mov    %r13,%rsi
    29ed:	44 89 e7             	mov    %r12d,%edi
    29f0:	e8 ab ef ff ff       	callq  19a0 <connect@plt>
    29f5:	85 c0                	test   %eax,%eax
    29f7:	0f 88 7d 01 00 00    	js     2b7a <submitr+0x261>
    29fd:	49 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%r9
    2a04:	b8 00 00 00 00       	mov    $0x0,%eax
    2a09:	4c 89 c9             	mov    %r9,%rcx
    2a0c:	48 89 df             	mov    %rbx,%rdi
    2a0f:	f2 ae                	repnz scas %es:(%rdi),%al
    2a11:	48 89 ce             	mov    %rcx,%rsi
    2a14:	48 f7 d6             	not    %rsi
    2a17:	4c 89 c9             	mov    %r9,%rcx
    2a1a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    2a1f:	f2 ae                	repnz scas %es:(%rdi),%al
    2a21:	49 89 c8             	mov    %rcx,%r8
    2a24:	4c 89 c9             	mov    %r9,%rcx
    2a27:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    2a2c:	f2 ae                	repnz scas %es:(%rdi),%al
    2a2e:	48 89 ca             	mov    %rcx,%rdx
    2a31:	48 f7 d2             	not    %rdx
    2a34:	4c 89 c9             	mov    %r9,%rcx
    2a37:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    2a3c:	f2 ae                	repnz scas %es:(%rdi),%al
    2a3e:	4c 29 c2             	sub    %r8,%rdx
    2a41:	48 29 ca             	sub    %rcx,%rdx
    2a44:	48 8d 44 76 fd       	lea    -0x3(%rsi,%rsi,2),%rax
    2a49:	48 8d 44 02 7b       	lea    0x7b(%rdx,%rax,1),%rax
    2a4e:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
    2a54:	0f 87 7d 01 00 00    	ja     2bd7 <submitr+0x2be>
    2a5a:	48 8d 94 24 60 40 00 	lea    0x4060(%rsp),%rdx
    2a61:	00 
    2a62:	b9 00 04 00 00       	mov    $0x400,%ecx
    2a67:	b8 00 00 00 00       	mov    $0x0,%eax
    2a6c:	48 89 d7             	mov    %rdx,%rdi
    2a6f:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    2a72:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
    2a79:	48 89 df             	mov    %rbx,%rdi
    2a7c:	f2 ae                	repnz scas %es:(%rdi),%al
    2a7e:	48 89 ca             	mov    %rcx,%rdx
    2a81:	48 f7 d2             	not    %rdx
    2a84:	48 89 d1             	mov    %rdx,%rcx
    2a87:	48 83 e9 01          	sub    $0x1,%rcx
    2a8b:	85 c9                	test   %ecx,%ecx
    2a8d:	0f 84 3f 06 00 00    	je     30d2 <submitr+0x7b9>
    2a93:	8d 41 ff             	lea    -0x1(%rcx),%eax
    2a96:	4c 8d 74 03 01       	lea    0x1(%rbx,%rax,1),%r14
    2a9b:	48 8d ac 24 60 40 00 	lea    0x4060(%rsp),%rbp
    2aa2:	00 
    2aa3:	48 8d 84 24 60 80 00 	lea    0x8060(%rsp),%rax
    2aaa:	00 
    2aab:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    2ab0:	49 bd d9 ff 00 00 00 	movabs $0x2000000000ffd9,%r13
    2ab7:	00 20 00 
    2aba:	e9 a6 01 00 00       	jmpq   2c65 <submitr+0x34c>
    2abf:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2ac6:	3a 20 43 
    2ac9:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2ad0:	20 75 6e 
    2ad3:	49 89 07             	mov    %rax,(%r15)
    2ad6:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2ada:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2ae1:	74 6f 20 
    2ae4:	48 ba 63 72 65 61 74 	movabs $0x7320657461657263,%rdx
    2aeb:	65 20 73 
    2aee:	49 89 47 10          	mov    %rax,0x10(%r15)
    2af2:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2af6:	41 c7 47 20 6f 63 6b 	movl   $0x656b636f,0x20(%r15)
    2afd:	65 
    2afe:	66 41 c7 47 24 74 00 	movw   $0x74,0x24(%r15)
    2b05:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2b0a:	e9 9a 04 00 00       	jmpq   2fa9 <submitr+0x690>
    2b0f:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
    2b16:	3a 20 44 
    2b19:	48 ba 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rdx
    2b20:	20 75 6e 
    2b23:	49 89 07             	mov    %rax,(%r15)
    2b26:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2b2a:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2b31:	74 6f 20 
    2b34:	48 ba 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rdx
    2b3b:	76 65 20 
    2b3e:	49 89 47 10          	mov    %rax,0x10(%r15)
    2b42:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2b46:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
    2b4d:	72 20 61 
    2b50:	49 89 47 20          	mov    %rax,0x20(%r15)
    2b54:	41 c7 47 28 64 64 72 	movl   $0x65726464,0x28(%r15)
    2b5b:	65 
    2b5c:	66 41 c7 47 2c 73 73 	movw   $0x7373,0x2c(%r15)
    2b63:	41 c6 47 2e 00       	movb   $0x0,0x2e(%r15)
    2b68:	44 89 e7             	mov    %r12d,%edi
    2b6b:	e8 60 ed ff ff       	callq  18d0 <close@plt>
    2b70:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2b75:	e9 2f 04 00 00       	jmpq   2fa9 <submitr+0x690>
    2b7a:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
    2b81:	3a 20 55 
    2b84:	48 ba 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rdx
    2b8b:	20 74 6f 
    2b8e:	49 89 07             	mov    %rax,(%r15)
    2b91:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2b95:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
    2b9c:	65 63 74 
    2b9f:	48 ba 20 74 6f 20 74 	movabs $0x20656874206f7420,%rdx
    2ba6:	68 65 20 
    2ba9:	49 89 47 10          	mov    %rax,0x10(%r15)
    2bad:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2bb1:	41 c7 47 20 73 65 72 	movl   $0x76726573,0x20(%r15)
    2bb8:	76 
    2bb9:	66 41 c7 47 24 65 72 	movw   $0x7265,0x24(%r15)
    2bc0:	41 c6 47 26 00       	movb   $0x0,0x26(%r15)
    2bc5:	44 89 e7             	mov    %r12d,%edi
    2bc8:	e8 03 ed ff ff       	callq  18d0 <close@plt>
    2bcd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2bd2:	e9 d2 03 00 00       	jmpq   2fa9 <submitr+0x690>
    2bd7:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
    2bde:	3a 20 52 
    2be1:	48 ba 65 73 75 6c 74 	movabs $0x747320746c757365,%rdx
    2be8:	20 73 74 
    2beb:	49 89 07             	mov    %rax,(%r15)
    2bee:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2bf2:	48 b8 72 69 6e 67 20 	movabs $0x6f6f7420676e6972,%rax
    2bf9:	74 6f 6f 
    2bfc:	48 ba 20 6c 61 72 67 	movabs $0x202e656772616c20,%rdx
    2c03:	65 2e 20 
    2c06:	49 89 47 10          	mov    %rax,0x10(%r15)
    2c0a:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2c0e:	48 b8 49 6e 63 72 65 	movabs $0x6573616572636e49,%rax
    2c15:	61 73 65 
    2c18:	48 ba 20 53 55 42 4d 	movabs $0x5254494d42555320,%rdx
    2c1f:	49 54 52 
    2c22:	49 89 47 20          	mov    %rax,0x20(%r15)
    2c26:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2c2a:	48 b8 5f 4d 41 58 42 	movabs $0x46554258414d5f,%rax
    2c31:	55 46 00 
    2c34:	49 89 47 30          	mov    %rax,0x30(%r15)
    2c38:	44 89 e7             	mov    %r12d,%edi
    2c3b:	e8 90 ec ff ff       	callq  18d0 <close@plt>
    2c40:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2c45:	e9 5f 03 00 00       	jmpq   2fa9 <submitr+0x690>
    2c4a:	49 0f a3 c5          	bt     %rax,%r13
    2c4e:	73 21                	jae    2c71 <submitr+0x358>
    2c50:	44 88 45 00          	mov    %r8b,0x0(%rbp)
    2c54:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
    2c58:	48 83 c3 01          	add    $0x1,%rbx
    2c5c:	4c 39 f3             	cmp    %r14,%rbx
    2c5f:	0f 84 6d 04 00 00    	je     30d2 <submitr+0x7b9>
    2c65:	44 0f b6 03          	movzbl (%rbx),%r8d
    2c69:	41 8d 40 d6          	lea    -0x2a(%r8),%eax
    2c6d:	3c 35                	cmp    $0x35,%al
    2c6f:	76 d9                	jbe    2c4a <submitr+0x331>
    2c71:	44 89 c0             	mov    %r8d,%eax
    2c74:	83 e0 df             	and    $0xffffffdf,%eax
    2c77:	83 e8 41             	sub    $0x41,%eax
    2c7a:	3c 19                	cmp    $0x19,%al
    2c7c:	76 d2                	jbe    2c50 <submitr+0x337>
    2c7e:	41 80 f8 20          	cmp    $0x20,%r8b
    2c82:	74 60                	je     2ce4 <submitr+0x3cb>
    2c84:	41 8d 40 e0          	lea    -0x20(%r8),%eax
    2c88:	3c 5f                	cmp    $0x5f,%al
    2c8a:	76 0a                	jbe    2c96 <submitr+0x37d>
    2c8c:	41 80 f8 09          	cmp    $0x9,%r8b
    2c90:	0f 85 af 03 00 00    	jne    3045 <submitr+0x72c>
    2c96:	45 0f b6 c0          	movzbl %r8b,%r8d
    2c9a:	48 8d 0d ef 10 00 00 	lea    0x10ef(%rip),%rcx        # 3d90 <array.3433+0x710>
    2ca1:	ba 08 00 00 00       	mov    $0x8,%edx
    2ca6:	be 01 00 00 00       	mov    $0x1,%esi
    2cab:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2cb0:	b8 00 00 00 00       	mov    $0x0,%eax
    2cb5:	e8 26 ed ff ff       	callq  19e0 <__sprintf_chk@plt>
    2cba:	0f b6 84 24 60 80 00 	movzbl 0x8060(%rsp),%eax
    2cc1:	00 
    2cc2:	88 45 00             	mov    %al,0x0(%rbp)
    2cc5:	0f b6 84 24 61 80 00 	movzbl 0x8061(%rsp),%eax
    2ccc:	00 
    2ccd:	88 45 01             	mov    %al,0x1(%rbp)
    2cd0:	0f b6 84 24 62 80 00 	movzbl 0x8062(%rsp),%eax
    2cd7:	00 
    2cd8:	88 45 02             	mov    %al,0x2(%rbp)
    2cdb:	48 8d 6d 03          	lea    0x3(%rbp),%rbp
    2cdf:	e9 74 ff ff ff       	jmpq   2c58 <submitr+0x33f>
    2ce4:	c6 45 00 2b          	movb   $0x2b,0x0(%rbp)
    2ce8:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
    2cec:	e9 67 ff ff ff       	jmpq   2c58 <submitr+0x33f>
    2cf1:	49 01 c5             	add    %rax,%r13
    2cf4:	48 29 c5             	sub    %rax,%rbp
    2cf7:	74 26                	je     2d1f <submitr+0x406>
    2cf9:	48 89 ea             	mov    %rbp,%rdx
    2cfc:	4c 89 ee             	mov    %r13,%rsi
    2cff:	44 89 e7             	mov    %r12d,%edi
    2d02:	e8 99 eb ff ff       	callq  18a0 <write@plt>
    2d07:	48 85 c0             	test   %rax,%rax
    2d0a:	7f e5                	jg     2cf1 <submitr+0x3d8>
    2d0c:	e8 5f eb ff ff       	callq  1870 <__errno_location@plt>
    2d11:	83 38 04             	cmpl   $0x4,(%rax)
    2d14:	0f 85 31 01 00 00    	jne    2e4b <submitr+0x532>
    2d1a:	4c 89 f0             	mov    %r14,%rax
    2d1d:	eb d2                	jmp    2cf1 <submitr+0x3d8>
    2d1f:	48 85 db             	test   %rbx,%rbx
    2d22:	0f 88 23 01 00 00    	js     2e4b <submitr+0x532>
    2d28:	44 89 64 24 50       	mov    %r12d,0x50(%rsp)
    2d2d:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
    2d34:	00 
    2d35:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
    2d3a:	48 8d 47 10          	lea    0x10(%rdi),%rax
    2d3e:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    2d43:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    2d4a:	00 
    2d4b:	ba 00 20 00 00       	mov    $0x2000,%edx
    2d50:	e8 08 fb ff ff       	callq  285d <rio_readlineb>
    2d55:	48 85 c0             	test   %rax,%rax
    2d58:	0f 8e 4c 01 00 00    	jle    2eaa <submitr+0x591>
    2d5e:	48 8d 4c 24 3c       	lea    0x3c(%rsp),%rcx
    2d63:	48 8d 94 24 60 60 00 	lea    0x6060(%rsp),%rdx
    2d6a:	00 
    2d6b:	48 8d bc 24 60 20 00 	lea    0x2060(%rsp),%rdi
    2d72:	00 
    2d73:	4c 8d 84 24 60 80 00 	lea    0x8060(%rsp),%r8
    2d7a:	00 
    2d7b:	48 8d 35 15 10 00 00 	lea    0x1015(%rip),%rsi        # 3d97 <array.3433+0x717>
    2d82:	b8 00 00 00 00       	mov    $0x0,%eax
    2d87:	e8 c4 eb ff ff       	callq  1950 <__isoc99_sscanf@plt>
    2d8c:	44 8b 44 24 3c       	mov    0x3c(%rsp),%r8d
    2d91:	41 81 f8 c8 00 00 00 	cmp    $0xc8,%r8d
    2d98:	0f 85 80 01 00 00    	jne    2f1e <submitr+0x605>
    2d9e:	48 8d 9c 24 60 20 00 	lea    0x2060(%rsp),%rbx
    2da5:	00 
    2da6:	48 8d 2d fb 0f 00 00 	lea    0xffb(%rip),%rbp        # 3da8 <array.3433+0x728>
    2dad:	4c 8d 6c 24 50       	lea    0x50(%rsp),%r13
    2db2:	b9 03 00 00 00       	mov    $0x3,%ecx
    2db7:	48 89 de             	mov    %rbx,%rsi
    2dba:	48 89 ef             	mov    %rbp,%rdi
    2dbd:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
    2dbf:	0f 97 c0             	seta   %al
    2dc2:	1c 00                	sbb    $0x0,%al
    2dc4:	84 c0                	test   %al,%al
    2dc6:	0f 84 89 01 00 00    	je     2f55 <submitr+0x63c>
    2dcc:	ba 00 20 00 00       	mov    $0x2000,%edx
    2dd1:	48 89 de             	mov    %rbx,%rsi
    2dd4:	4c 89 ef             	mov    %r13,%rdi
    2dd7:	e8 81 fa ff ff       	callq  285d <rio_readlineb>
    2ddc:	48 85 c0             	test   %rax,%rax
    2ddf:	7f d1                	jg     2db2 <submitr+0x499>
    2de1:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2de8:	3a 20 43 
    2deb:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2df2:	20 75 6e 
    2df5:	49 89 07             	mov    %rax,(%r15)
    2df8:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2dfc:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2e03:	74 6f 20 
    2e06:	48 ba 72 65 61 64 20 	movabs $0x6165682064616572,%rdx
    2e0d:	68 65 61 
    2e10:	49 89 47 10          	mov    %rax,0x10(%r15)
    2e14:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2e18:	48 b8 64 65 72 73 20 	movabs $0x6f72662073726564,%rax
    2e1f:	66 72 6f 
    2e22:	48 ba 6d 20 73 65 72 	movabs $0x726576726573206d,%rdx
    2e29:	76 65 72 
    2e2c:	49 89 47 20          	mov    %rax,0x20(%r15)
    2e30:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2e34:	41 c6 47 30 00       	movb   $0x0,0x30(%r15)
    2e39:	44 89 e7             	mov    %r12d,%edi
    2e3c:	e8 8f ea ff ff       	callq  18d0 <close@plt>
    2e41:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2e46:	e9 5e 01 00 00       	jmpq   2fa9 <submitr+0x690>
    2e4b:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2e52:	3a 20 43 
    2e55:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2e5c:	20 75 6e 
    2e5f:	49 89 07             	mov    %rax,(%r15)
    2e62:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2e66:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2e6d:	74 6f 20 
    2e70:	48 ba 77 72 69 74 65 	movabs $0x6f74206574697277,%rdx
    2e77:	20 74 6f 
    2e7a:	49 89 47 10          	mov    %rax,0x10(%r15)
    2e7e:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2e82:	48 b8 20 74 68 65 20 	movabs $0x7265732065687420,%rax
    2e89:	73 65 72 
    2e8c:	49 89 47 20          	mov    %rax,0x20(%r15)
    2e90:	41 c7 47 28 76 65 72 	movl   $0x726576,0x28(%r15)
    2e97:	00 
    2e98:	44 89 e7             	mov    %r12d,%edi
    2e9b:	e8 30 ea ff ff       	callq  18d0 <close@plt>
    2ea0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2ea5:	e9 ff 00 00 00       	jmpq   2fa9 <submitr+0x690>
    2eaa:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2eb1:	3a 20 43 
    2eb4:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2ebb:	20 75 6e 
    2ebe:	49 89 07             	mov    %rax,(%r15)
    2ec1:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2ec5:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2ecc:	74 6f 20 
    2ecf:	48 ba 72 65 61 64 20 	movabs $0x7269662064616572,%rdx
    2ed6:	66 69 72 
    2ed9:	49 89 47 10          	mov    %rax,0x10(%r15)
    2edd:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2ee1:	48 b8 73 74 20 68 65 	movabs $0x6564616568207473,%rax
    2ee8:	61 64 65 
    2eeb:	48 ba 72 20 66 72 6f 	movabs $0x73206d6f72662072,%rdx
    2ef2:	6d 20 73 
    2ef5:	49 89 47 20          	mov    %rax,0x20(%r15)
    2ef9:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2efd:	41 c7 47 30 65 72 76 	movl   $0x65767265,0x30(%r15)
    2f04:	65 
    2f05:	66 41 c7 47 34 72 00 	movw   $0x72,0x34(%r15)
    2f0c:	44 89 e7             	mov    %r12d,%edi
    2f0f:	e8 bc e9 ff ff       	callq  18d0 <close@plt>
    2f14:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2f19:	e9 8b 00 00 00       	jmpq   2fa9 <submitr+0x690>
    2f1e:	4c 8d 8c 24 60 80 00 	lea    0x8060(%rsp),%r9
    2f25:	00 
    2f26:	48 8d 0d b3 0d 00 00 	lea    0xdb3(%rip),%rcx        # 3ce0 <array.3433+0x660>
    2f2d:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
    2f34:	be 01 00 00 00       	mov    $0x1,%esi
    2f39:	4c 89 ff             	mov    %r15,%rdi
    2f3c:	b8 00 00 00 00       	mov    $0x0,%eax
    2f41:	e8 9a ea ff ff       	callq  19e0 <__sprintf_chk@plt>
    2f46:	44 89 e7             	mov    %r12d,%edi
    2f49:	e8 82 e9 ff ff       	callq  18d0 <close@plt>
    2f4e:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2f53:	eb 54                	jmp    2fa9 <submitr+0x690>
    2f55:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    2f5c:	00 
    2f5d:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
    2f62:	ba 00 20 00 00       	mov    $0x2000,%edx
    2f67:	e8 f1 f8 ff ff       	callq  285d <rio_readlineb>
    2f6c:	48 85 c0             	test   %rax,%rax
    2f6f:	7e 61                	jle    2fd2 <submitr+0x6b9>
    2f71:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    2f78:	00 
    2f79:	4c 89 ff             	mov    %r15,%rdi
    2f7c:	e8 ff e8 ff ff       	callq  1880 <strcpy@plt>
    2f81:	44 89 e7             	mov    %r12d,%edi
    2f84:	e8 47 e9 ff ff       	callq  18d0 <close@plt>
    2f89:	b9 03 00 00 00       	mov    $0x3,%ecx
    2f8e:	48 8d 3d 16 0e 00 00 	lea    0xe16(%rip),%rdi        # 3dab <array.3433+0x72b>
    2f95:	4c 89 fe             	mov    %r15,%rsi
    2f98:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
    2f9a:	0f 97 c0             	seta   %al
    2f9d:	1c 00                	sbb    $0x0,%al
    2f9f:	84 c0                	test   %al,%al
    2fa1:	0f 95 c0             	setne  %al
    2fa4:	0f b6 c0             	movzbl %al,%eax
    2fa7:	f7 d8                	neg    %eax
    2fa9:	48 8b 94 24 68 a0 00 	mov    0xa068(%rsp),%rdx
    2fb0:	00 
    2fb1:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
    2fb8:	00 00 
    2fba:	0f 85 95 01 00 00    	jne    3155 <submitr+0x83c>
    2fc0:	48 81 c4 78 a0 00 00 	add    $0xa078,%rsp
    2fc7:	5b                   	pop    %rbx
    2fc8:	5d                   	pop    %rbp
    2fc9:	41 5c                	pop    %r12
    2fcb:	41 5d                	pop    %r13
    2fcd:	41 5e                	pop    %r14
    2fcf:	41 5f                	pop    %r15
    2fd1:	c3                   	retq   
    2fd2:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2fd9:	3a 20 43 
    2fdc:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2fe3:	20 75 6e 
    2fe6:	49 89 07             	mov    %rax,(%r15)
    2fe9:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2fed:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2ff4:	74 6f 20 
    2ff7:	48 ba 72 65 61 64 20 	movabs $0x6174732064616572,%rdx
    2ffe:	73 74 61 
    3001:	49 89 47 10          	mov    %rax,0x10(%r15)
    3005:	49 89 57 18          	mov    %rdx,0x18(%r15)
    3009:	48 b8 74 75 73 20 6d 	movabs $0x7373656d20737574,%rax
    3010:	65 73 73 
    3013:	48 ba 61 67 65 20 66 	movabs $0x6d6f726620656761,%rdx
    301a:	72 6f 6d 
    301d:	49 89 47 20          	mov    %rax,0x20(%r15)
    3021:	49 89 57 28          	mov    %rdx,0x28(%r15)
    3025:	48 b8 20 73 65 72 76 	movabs $0x72657672657320,%rax
    302c:	65 72 00 
    302f:	49 89 47 30          	mov    %rax,0x30(%r15)
    3033:	44 89 e7             	mov    %r12d,%edi
    3036:	e8 95 e8 ff ff       	callq  18d0 <close@plt>
    303b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    3040:	e9 64 ff ff ff       	jmpq   2fa9 <submitr+0x690>
    3045:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
    304c:	3a 20 52 
    304f:	48 ba 65 73 75 6c 74 	movabs $0x747320746c757365,%rdx
    3056:	20 73 74 
    3059:	49 89 07             	mov    %rax,(%r15)
    305c:	49 89 57 08          	mov    %rdx,0x8(%r15)
    3060:	48 b8 72 69 6e 67 20 	movabs $0x6e6f6320676e6972,%rax
    3067:	63 6f 6e 
    306a:	48 ba 74 61 69 6e 73 	movabs $0x6e6120736e696174,%rdx
    3071:	20 61 6e 
    3074:	49 89 47 10          	mov    %rax,0x10(%r15)
    3078:	49 89 57 18          	mov    %rdx,0x18(%r15)
    307c:	48 b8 20 69 6c 6c 65 	movabs $0x6c6167656c6c6920,%rax
    3083:	67 61 6c 
    3086:	48 ba 20 6f 72 20 75 	movabs $0x72706e7520726f20,%rdx
    308d:	6e 70 72 
    3090:	49 89 47 20          	mov    %rax,0x20(%r15)
    3094:	49 89 57 28          	mov    %rdx,0x28(%r15)
    3098:	48 b8 69 6e 74 61 62 	movabs $0x20656c6261746e69,%rax
    309f:	6c 65 20 
    30a2:	48 ba 63 68 61 72 61 	movabs $0x6574636172616863,%rdx
    30a9:	63 74 65 
    30ac:	49 89 47 30          	mov    %rax,0x30(%r15)
    30b0:	49 89 57 38          	mov    %rdx,0x38(%r15)
    30b4:	66 41 c7 47 40 72 2e 	movw   $0x2e72,0x40(%r15)
    30bb:	41 c6 47 42 00       	movb   $0x0,0x42(%r15)
    30c0:	44 89 e7             	mov    %r12d,%edi
    30c3:	e8 08 e8 ff ff       	callq  18d0 <close@plt>
    30c8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    30cd:	e9 d7 fe ff ff       	jmpq   2fa9 <submitr+0x690>
    30d2:	48 8d 9c 24 60 20 00 	lea    0x2060(%rsp),%rbx
    30d9:	00 
    30da:	48 83 ec 08          	sub    $0x8,%rsp
    30de:	48 8d 84 24 68 40 00 	lea    0x4068(%rsp),%rax
    30e5:	00 
    30e6:	50                   	push   %rax
    30e7:	ff 74 24 28          	pushq  0x28(%rsp)
    30eb:	ff 74 24 38          	pushq  0x38(%rsp)
    30ef:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
    30f4:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    30f9:	48 8d 0d 10 0c 00 00 	lea    0xc10(%rip),%rcx        # 3d10 <array.3433+0x690>
    3100:	ba 00 20 00 00       	mov    $0x2000,%edx
    3105:	be 01 00 00 00       	mov    $0x1,%esi
    310a:	48 89 df             	mov    %rbx,%rdi
    310d:	b8 00 00 00 00       	mov    $0x0,%eax
    3112:	e8 c9 e8 ff ff       	callq  19e0 <__sprintf_chk@plt>
    3117:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
    311e:	b8 00 00 00 00       	mov    $0x0,%eax
    3123:	48 89 df             	mov    %rbx,%rdi
    3126:	f2 ae                	repnz scas %es:(%rdi),%al
    3128:	48 f7 d1             	not    %rcx
    312b:	48 89 cb             	mov    %rcx,%rbx
    312e:	48 83 eb 01          	sub    $0x1,%rbx
    3132:	48 83 c4 20          	add    $0x20,%rsp
    3136:	48 89 dd             	mov    %rbx,%rbp
    3139:	4c 8d ac 24 60 20 00 	lea    0x2060(%rsp),%r13
    3140:	00 
    3141:	41 be 00 00 00 00    	mov    $0x0,%r14d
    3147:	48 85 db             	test   %rbx,%rbx
    314a:	0f 85 a9 fb ff ff    	jne    2cf9 <submitr+0x3e0>
    3150:	e9 d3 fb ff ff       	jmpq   2d28 <submitr+0x40f>
    3155:	e8 56 e7 ff ff       	callq  18b0 <__stack_chk_fail@plt>

000000000000315a <init_timeout>:
    315a:	85 ff                	test   %edi,%edi
    315c:	74 25                	je     3183 <init_timeout+0x29>
    315e:	53                   	push   %rbx
    315f:	89 fb                	mov    %edi,%ebx
    3161:	48 8d 35 c5 f6 ff ff 	lea    -0x93b(%rip),%rsi        # 282d <sigalrm_handler>
    3168:	bf 0e 00 00 00       	mov    $0xe,%edi
    316d:	e8 8e e7 ff ff       	callq  1900 <signal@plt>
    3172:	85 db                	test   %ebx,%ebx
    3174:	bf 00 00 00 00       	mov    $0x0,%edi
    3179:	0f 49 fb             	cmovns %ebx,%edi
    317c:	e8 3f e7 ff ff       	callq  18c0 <alarm@plt>
    3181:	5b                   	pop    %rbx
    3182:	c3                   	retq   
    3183:	f3 c3                	repz retq 

0000000000003185 <init_driver>:
    3185:	41 54                	push   %r12
    3187:	55                   	push   %rbp
    3188:	53                   	push   %rbx
    3189:	48 83 ec 20          	sub    $0x20,%rsp
    318d:	49 89 fc             	mov    %rdi,%r12
    3190:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3197:	00 00 
    3199:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    319e:	31 c0                	xor    %eax,%eax
    31a0:	be 01 00 00 00       	mov    $0x1,%esi
    31a5:	bf 0d 00 00 00       	mov    $0xd,%edi
    31aa:	e8 51 e7 ff ff       	callq  1900 <signal@plt>
    31af:	be 01 00 00 00       	mov    $0x1,%esi
    31b4:	bf 1d 00 00 00       	mov    $0x1d,%edi
    31b9:	e8 42 e7 ff ff       	callq  1900 <signal@plt>
    31be:	be 01 00 00 00       	mov    $0x1,%esi
    31c3:	bf 1d 00 00 00       	mov    $0x1d,%edi
    31c8:	e8 33 e7 ff ff       	callq  1900 <signal@plt>
    31cd:	ba 00 00 00 00       	mov    $0x0,%edx
    31d2:	be 01 00 00 00       	mov    $0x1,%esi
    31d7:	bf 02 00 00 00       	mov    $0x2,%edi
    31dc:	e8 0f e8 ff ff       	callq  19f0 <socket@plt>
    31e1:	85 c0                	test   %eax,%eax
    31e3:	0f 88 a3 00 00 00    	js     328c <init_driver+0x107>
    31e9:	89 c3                	mov    %eax,%ebx
    31eb:	48 8d 3d bc 0b 00 00 	lea    0xbbc(%rip),%rdi        # 3dae <array.3433+0x72e>
    31f2:	e8 19 e7 ff ff       	callq  1910 <gethostbyname@plt>
    31f7:	48 85 c0             	test   %rax,%rax
    31fa:	0f 84 df 00 00 00    	je     32df <init_driver+0x15a>
    3200:	48 89 e5             	mov    %rsp,%rbp
    3203:	48 c7 44 24 02 00 00 	movq   $0x0,0x2(%rsp)
    320a:	00 00 
    320c:	c7 45 0a 00 00 00 00 	movl   $0x0,0xa(%rbp)
    3213:	66 c7 45 0e 00 00    	movw   $0x0,0xe(%rbp)
    3219:	66 c7 04 24 02 00    	movw   $0x2,(%rsp)
    321f:	48 63 50 14          	movslq 0x14(%rax),%rdx
    3223:	48 8b 40 18          	mov    0x18(%rax),%rax
    3227:	48 8d 7d 04          	lea    0x4(%rbp),%rdi
    322b:	b9 0c 00 00 00       	mov    $0xc,%ecx
    3230:	48 8b 30             	mov    (%rax),%rsi
    3233:	e8 e8 e6 ff ff       	callq  1920 <__memmove_chk@plt>
    3238:	66 c7 44 24 02 3c 9a 	movw   $0x9a3c,0x2(%rsp)
    323f:	ba 10 00 00 00       	mov    $0x10,%edx
    3244:	48 89 ee             	mov    %rbp,%rsi
    3247:	89 df                	mov    %ebx,%edi
    3249:	e8 52 e7 ff ff       	callq  19a0 <connect@plt>
    324e:	85 c0                	test   %eax,%eax
    3250:	0f 88 fb 00 00 00    	js     3351 <init_driver+0x1cc>
    3256:	89 df                	mov    %ebx,%edi
    3258:	e8 73 e6 ff ff       	callq  18d0 <close@plt>
    325d:	66 41 c7 04 24 4f 4b 	movw   $0x4b4f,(%r12)
    3264:	41 c6 44 24 02 00    	movb   $0x0,0x2(%r12)
    326a:	b8 00 00 00 00       	mov    $0x0,%eax
    326f:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    3274:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
    327b:	00 00 
    327d:	0f 85 06 01 00 00    	jne    3389 <init_driver+0x204>
    3283:	48 83 c4 20          	add    $0x20,%rsp
    3287:	5b                   	pop    %rbx
    3288:	5d                   	pop    %rbp
    3289:	41 5c                	pop    %r12
    328b:	c3                   	retq   
    328c:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    3293:	3a 20 43 
    3296:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    329d:	20 75 6e 
    32a0:	49 89 04 24          	mov    %rax,(%r12)
    32a4:	49 89 54 24 08       	mov    %rdx,0x8(%r12)
    32a9:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    32b0:	74 6f 20 
    32b3:	48 ba 63 72 65 61 74 	movabs $0x7320657461657263,%rdx
    32ba:	65 20 73 
    32bd:	49 89 44 24 10       	mov    %rax,0x10(%r12)
    32c2:	49 89 54 24 18       	mov    %rdx,0x18(%r12)
    32c7:	41 c7 44 24 20 6f 63 	movl   $0x656b636f,0x20(%r12)
    32ce:	6b 65 
    32d0:	66 41 c7 44 24 24 74 	movw   $0x74,0x24(%r12)
    32d7:	00 
    32d8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    32dd:	eb 90                	jmp    326f <init_driver+0xea>
    32df:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
    32e6:	3a 20 44 
    32e9:	48 ba 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rdx
    32f0:	20 75 6e 
    32f3:	49 89 04 24          	mov    %rax,(%r12)
    32f7:	49 89 54 24 08       	mov    %rdx,0x8(%r12)
    32fc:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    3303:	74 6f 20 
    3306:	48 ba 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rdx
    330d:	76 65 20 
    3310:	49 89 44 24 10       	mov    %rax,0x10(%r12)
    3315:	49 89 54 24 18       	mov    %rdx,0x18(%r12)
    331a:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
    3321:	72 20 61 
    3324:	49 89 44 24 20       	mov    %rax,0x20(%r12)
    3329:	41 c7 44 24 28 64 64 	movl   $0x65726464,0x28(%r12)
    3330:	72 65 
    3332:	66 41 c7 44 24 2c 73 	movw   $0x7373,0x2c(%r12)
    3339:	73 
    333a:	41 c6 44 24 2e 00    	movb   $0x0,0x2e(%r12)
    3340:	89 df                	mov    %ebx,%edi
    3342:	e8 89 e5 ff ff       	callq  18d0 <close@plt>
    3347:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    334c:	e9 1e ff ff ff       	jmpq   326f <init_driver+0xea>
    3351:	4c 8d 05 56 0a 00 00 	lea    0xa56(%rip),%r8        # 3dae <array.3433+0x72e>
    3358:	48 8d 0d 09 0a 00 00 	lea    0xa09(%rip),%rcx        # 3d68 <array.3433+0x6e8>
    335f:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
    3366:	be 01 00 00 00       	mov    $0x1,%esi
    336b:	4c 89 e7             	mov    %r12,%rdi
    336e:	b8 00 00 00 00       	mov    $0x0,%eax
    3373:	e8 68 e6 ff ff       	callq  19e0 <__sprintf_chk@plt>
    3378:	89 df                	mov    %ebx,%edi
    337a:	e8 51 e5 ff ff       	callq  18d0 <close@plt>
    337f:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    3384:	e9 e6 fe ff ff       	jmpq   326f <init_driver+0xea>
    3389:	e8 22 e5 ff ff       	callq  18b0 <__stack_chk_fail@plt>

000000000000338e <driver_post>:
    338e:	53                   	push   %rbx
    338f:	4c 89 c3             	mov    %r8,%rbx
    3392:	85 c9                	test   %ecx,%ecx
    3394:	75 17                	jne    33ad <driver_post+0x1f>
    3396:	48 85 ff             	test   %rdi,%rdi
    3399:	74 05                	je     33a0 <driver_post+0x12>
    339b:	80 3f 00             	cmpb   $0x0,(%rdi)
    339e:	75 33                	jne    33d3 <driver_post+0x45>
    33a0:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
    33a5:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
    33a9:	89 c8                	mov    %ecx,%eax
    33ab:	5b                   	pop    %rbx
    33ac:	c3                   	retq   
    33ad:	48 8d 35 13 0a 00 00 	lea    0xa13(%rip),%rsi        # 3dc7 <array.3433+0x747>
    33b4:	bf 01 00 00 00       	mov    $0x1,%edi
    33b9:	b8 00 00 00 00       	mov    $0x0,%eax
    33be:	e8 9d e5 ff ff       	callq  1960 <__printf_chk@plt>
    33c3:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
    33c8:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
    33cc:	b8 00 00 00 00       	mov    $0x0,%eax
    33d1:	eb d8                	jmp    33ab <driver_post+0x1d>
    33d3:	41 50                	push   %r8
    33d5:	52                   	push   %rdx
    33d6:	4c 8d 0d 01 0a 00 00 	lea    0xa01(%rip),%r9        # 3dde <array.3433+0x75e>
    33dd:	49 89 f0             	mov    %rsi,%r8
    33e0:	48 89 f9             	mov    %rdi,%rcx
    33e3:	48 8d 15 f8 09 00 00 	lea    0x9f8(%rip),%rdx        # 3de2 <array.3433+0x762>
    33ea:	be 9a 3c 00 00       	mov    $0x3c9a,%esi
    33ef:	48 8d 3d b8 09 00 00 	lea    0x9b8(%rip),%rdi        # 3dae <array.3433+0x72e>
    33f6:	e8 1e f5 ff ff       	callq  2919 <submitr>
    33fb:	48 83 c4 10          	add    $0x10,%rsp
    33ff:	eb aa                	jmp    33ab <driver_post+0x1d>
    3401:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    3408:	00 00 00 
    340b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000003410 <__libc_csu_init>:
    3410:	41 57                	push   %r15
    3412:	41 56                	push   %r14
    3414:	49 89 d7             	mov    %rdx,%r15
    3417:	41 55                	push   %r13
    3419:	41 54                	push   %r12
    341b:	4c 8d 25 be 18 20 00 	lea    0x2018be(%rip),%r12        # 204ce0 <__frame_dummy_init_array_entry>
    3422:	55                   	push   %rbp
    3423:	48 8d 2d be 18 20 00 	lea    0x2018be(%rip),%rbp        # 204ce8 <__init_array_end>
    342a:	53                   	push   %rbx
    342b:	41 89 fd             	mov    %edi,%r13d
    342e:	49 89 f6             	mov    %rsi,%r14
    3431:	4c 29 e5             	sub    %r12,%rbp
    3434:	48 83 ec 08          	sub    $0x8,%rsp
    3438:	48 c1 fd 03          	sar    $0x3,%rbp
    343c:	e8 d7 e3 ff ff       	callq  1818 <_init>
    3441:	48 85 ed             	test   %rbp,%rbp
    3444:	74 20                	je     3466 <__libc_csu_init+0x56>
    3446:	31 db                	xor    %ebx,%ebx
    3448:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    344f:	00 
    3450:	4c 89 fa             	mov    %r15,%rdx
    3453:	4c 89 f6             	mov    %r14,%rsi
    3456:	44 89 ef             	mov    %r13d,%edi
    3459:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
    345d:	48 83 c3 01          	add    $0x1,%rbx
    3461:	48 39 dd             	cmp    %rbx,%rbp
    3464:	75 ea                	jne    3450 <__libc_csu_init+0x40>
    3466:	48 83 c4 08          	add    $0x8,%rsp
    346a:	5b                   	pop    %rbx
    346b:	5d                   	pop    %rbp
    346c:	41 5c                	pop    %r12
    346e:	41 5d                	pop    %r13
    3470:	41 5e                	pop    %r14
    3472:	41 5f                	pop    %r15
    3474:	c3                   	retq   
    3475:	90                   	nop
    3476:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    347d:	00 00 00 

0000000000003480 <__libc_csu_fini>:
    3480:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000003484 <_fini>:
    3484:	48 83 ec 08          	sub    $0x8,%rsp
    3488:	48 83 c4 08          	add    $0x8,%rsp
    348c:	c3                   	retq   
