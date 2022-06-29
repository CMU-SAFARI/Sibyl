#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int openFastDevice()
{
	int fp = open("/dev/nvme0n1", O_RDWR|O_SYNC);
    printf("FP=%d\n", fp);
    if(fp <= 0) {
        perror("Error opening file");
        return(-1);
    }
	return fp;
}

int openMiddleDevice()
{
	int fp = open("/dev/sdb", O_RDWR|O_SYNC);
    printf("FP=%d\n", fp);
    if(fp <= 0) {
        perror("Error opening file");
        return(-1);
    }
	return fp;
}

int openSlowDevice()
{
	int fp = open("/dev/sda3", O_RDWR|O_SYNC);
    printf("FP=%d\n", fp);
    if(fp <= 0) {
        perror("Error opening file");
        return(-1);
    }
	return fp;
}

int sibyl_read(int fd, unsigned long byte_offset, unsigned int nSize)
{
	char readBuf[nSize] __attribute__ ((__aligned__ (4096)));
	memset(readBuf, 0x00, sizeof(char) * nSize);
	off_t readOffset = lseek(fd, byte_offset, SEEK_SET);
	ssize_t len = read(fd, readBuf, sizeof(char) * nSize);
	return len;
}

int sibyl_write(int fd, unsigned long byte_offset, unsigned int nSize)
{
	char writeBuf[nSize] __attribute__ ((__aligned__ (4096)));
	memset(writeBuf, 0xA5, sizeof(char) * nSize);
	off_t writeOffset = lseek(fd, byte_offset, SEEK_SET);
	ssize_t len = write(fd, writeBuf, sizeof(char) * nSize);
	return len;
}

void closeDevice(int fd)
{
	close(fd);
}
