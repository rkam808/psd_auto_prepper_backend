class ProcessingJobsController < ApplicationController
  # For disabling CSRF tokens on API controllers.
  # protect_from_forgery with: :null_session

  def create
    job = ProcessingJob.create(status: :pending)

    if params[:processing_job] && params[:processing_job][:original_image]
      job.original_image.attach(params[:processing_job][:original_image])
    end

    ::PsdGeneratorService.new(job.id).call

    job.reload

    render json: { id: job.id, status: job.status }
  end

  def show
    job = ProcessingJob.find(params[:id])

    if job.completed?
      render json: {
        id: job.id,
        status: job.status,
        prepped_psd_url: rails_blob_url(job.prepped_psd)
      }
    else
      render json: { id: job.id, status: job.status }
    end
  end
end
