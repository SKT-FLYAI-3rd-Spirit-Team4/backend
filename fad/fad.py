from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `PANN`
# frechet = FrechetAudioDistance(
#     model_name="pann",
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )
fad_score = frechet.score("D:/2023_PROJECT/backend/fad/classic-f", "D:/2023_PROJECT/backend/fad/classic-f", dtype="float32")
print(fad_score)