using DataDeps

register(DataDep(
    "WikiText-103",
    """
    WikiText Long Term Dependency Language Modeling Dataset
    This is a language modelling dataset under Creative Commons Attribution-ShareAlike License. I contains over 100
    million tokens.
    """,
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    post_fetch_method = function (file)
        unpack(file)
        dir = "wikitext-103"
        files = readdir(dir)
        mv.(joinpath.(dir, files), files)
        rm(dir)
    end
))
