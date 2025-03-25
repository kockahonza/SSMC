using Base.Threads
using ChunkSplitters


function kaka1(n=100)
    ret = fill(-1, n)
    for i in 1:n
        sleep(0.1 / n)
        ret[i] = threadid()
    end
    ret
end
function kaka2(n=100)
    ret = fill(-1, n)
    @threads for i in 1:n
        sleep(0.1 / n)
        ret[i] = threadid()
    end
    ret
end
function kaka3(n=100, k=nthreads())
    ret = fill(-1, n)

    ichunks = chunks(1:n, k)
    @sync for (is, _) in ichunks
        @spawn begin
            for i in is
                sleep(0.1 / n)
                ret[i] = threadid()
            end
        end
    end

    ret
end
