/* PATCHED (region-aware RL project).
 * Differences from stock ROSCO zmq_client.c:
 *   - 22 measurements instead of 17 (see ZeroMQInterface.f90)
 *   - one persistent REQ socket per process instead of connect/close every call
 *     (stock version pays a TCP handshake per control step; at 100 Hz that dominates)
 *   - received buffer is NUL-terminated from the actual message length
 *   - the whitespace stripper compares chars, not pointers (stock code compared s[i]==" ")
 *   - socket is closed when ROSCO signals its final call (iStatus == -1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>

#define NUM_MEAS 22
#define NUM_SETP 8
#define CHAR_SINGLE 24
#define CHAR_ARRAY (NUM_MEAS * (CHAR_SINGLE + 1))

static void *g_ctx = NULL;
static void *g_req = NULL;
static char  g_addr[256] = {0};

static void strip_blanks(char *s)
{
    char *d = s;
    for (; *s; ++s) if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r') *d++ = *s;
    *d = '\0';
}

static void close_socket(void)
{
    if (g_req) { zmq_close(g_req); g_req = NULL; }
    if (g_ctx) { zmq_ctx_destroy(g_ctx); g_ctx = NULL; }
    g_addr[0] = '\0';
}

int zmq_client(char *zmq_address, double measurements[NUM_MEAS], double setpoints[NUM_SETP])
{
    char to_ssc[CHAR_ARRAY];
    char from_ssc[CHAR_ARRAY];
    char b[CHAR_SINGLE];
    int i, n;

    /* (re)connect if address changed or first call */
    if (!g_req || strncmp(g_addr, zmq_address, sizeof(g_addr) - 1) != 0) {
        close_socket();
        g_ctx = zmq_ctx_new();
        g_req = zmq_socket(g_ctx, ZMQ_REQ);
        int linger = 0;
        zmq_setsockopt(g_req, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_connect(g_req, zmq_address);
        strncpy(g_addr, zmq_address, sizeof(g_addr) - 1);
    }

    to_ssc[0] = '\0';
    for (i = 0; i < NUM_MEAS; ++i) {
        snprintf(b, sizeof(b), "%s%.8e", i ? "," : "", measurements[i]);
        strncat(to_ssc, b, CHAR_ARRAY - strlen(to_ssc) - 1);
    }

    zmq_send(g_req, to_ssc, strlen(to_ssc), 0);
    n = zmq_recv(g_req, from_ssc, CHAR_ARRAY - 1, 0);
    if (n < 0) n = 0;
    if (n > CHAR_ARRAY - 1) n = CHAR_ARRAY - 1;
    from_ssc[n] = '\0';

    strip_blanks(from_ssc);
    for (i = 0; i < NUM_SETP; ++i) setpoints[i] = 0.0;
    char *pt = strtok(from_ssc, ",");
    i = 0;
    while (pt != NULL && i < NUM_SETP) {
        setpoints[i++] = atof(pt);
        pt = strtok(NULL, ",");
    }

    /* measurements[1] is ROSCO iStatus; -1 == final call */
    if (measurements[1] < -0.5) close_socket();
    return 0;
}
