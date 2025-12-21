class ProcessingJobsController < ApplicationController
  # For disabling CSRF tokens on API controllers.
  # protect_from_forgery with: :null_session

  def create
    job = ProcessingJob.create(status: :pending)
    job.original_image.attach(params[:image])

    render json: { id: job.id, status: job.status }
  end

  def show
    job = ProcessingJob.find(params[:id])

    if job.completed?
      render json: {
        id: job.id,
        status: job.status,
        prepped_psd_url: url_for(job.prepped_psd)
      }
    else
      render json: { id: job.id, status: job.status }
    end
  end
end
